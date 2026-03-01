# train_pickformer_v2.py
import os
import math
import random
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from util_new import ImageSliceDataset, DualCompose, DualResize 
from PickFormer_v2 import PickFormer, total_loss                

def save_checkpoint(epoch, model, optimizer, scheduler, scaler, path):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(ckpt, path)
    
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BinaryConfusionMeter:
    def __init__(self):
        self.tp = 0; self.fp = 0; self.fn = 0; self.tn = 0

    @torch.no_grad()
    def update(self, preds: torch.Tensor, masks: torch.Tensor):
        # preds/masks: [B,H,W] in {0,1} 或 [B,1,H,W]
        if preds.ndim == 4 and preds.shape[1] == 1:
            preds = preds.squeeze(1)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        preds = preds.long(); masks = masks.long()
        self.tp += int(((preds == 1) & (masks == 1)).sum().item())
        self.fp += int(((preds == 1) & (masks == 0)).sum().item())
        self.fn += int(((preds == 0) & (masks == 1)).sum().item())
        self.tn += int(((preds == 0) & (masks == 0)).sum().item())

    def safe_kappa(self) -> float:
        tp, fp, fn, tn = float(self.tp), float(self.fp), float(self.fn), float(self.tn)
        N = tp + fp + fn + tn
        if N == 0: return 0.0
        po = (tp + tn) / N
        p_yes_true = (tp + fn) / N; p_yes_pred = (tp + fp) / N
        p_no_true  = (fp + tn) / N; p_no_pred  = (fn + tn) / N
        pe = p_yes_true * p_yes_pred + p_no_true * p_no_pred
        denom = 1.0 - pe
        if abs(denom) < 1e-12:
            return 1.0 if abs(po - 1.0) < 1e-12 else 0.0
        return (po - pe) / denom

    def summary(self) -> dict:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-8
        iou_fg = tp / (tp + fp + fn + eps)
        iou_bg = tn / (tn + fn + fp + eps)
        dice_fg = 2*tp / (2*tp + fp + fn + eps)
        dice_bg = 2*tn / (2*tn + fn + fp + eps)
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1        = 2*precision*recall / (precision + recall + eps)
        acc       = (tp + tn) / (tp + fp + fn + tn + eps)
        return {
            "mIoU": float((iou_fg + iou_bg) / 2),
            "Dice": float((dice_fg + dice_bg) / 2),
            "F1-score": float(f1),
            "Pixel Acc": float(acc),
            "IoUs": [float(iou_fg), float(iou_bg)],
            "Dice Scores": [float(dice_fg), float(dice_bg)],
            "Kappa": float(self.safe_kappa())
        }


def get_main_logits(outputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs

import torch.nn.functional as F

def _fg_logits(logits, num_classes: int):

    return logits if logits.shape[1] == 1 or num_classes == 1 else logits[:, 1:2, ...]

def dice_loss_only(logits, targets, num_classes=2, eps=1e-6):
    L = _fg_logits(logits, num_classes)
    probs = torch.sigmoid(L)
    targets = targets.float()
    num = 2 * (probs * targets).sum(dim=(2, 3)) + eps
    den = (probs.pow(2) + targets.pow(2)).sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()

def bce_edge_combo(logits, targets, num_classes=2, pos_weight=None, w_edge=0.2):
    L = _fg_logits(logits, num_classes)
    bce = F.binary_cross_entropy_with_logits(L, targets.float(), pos_weight=pos_weight)
    # edge
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=logits.device).view(1,1,3,3)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=logits.device).view(1,1,3,3)
    probs = torch.sigmoid(L)
    gx_p = F.conv2d(probs, kx, padding=1) + F.conv2d(probs, ky, padding=1)
    gx_t = F.conv2d(targets.float(), kx, padding=1) + F.conv2d(targets.float(), ky, padding=1)
    edge = F.l1_loss(gx_p, gx_t)
    return bce + w_edge * edge

def single_head_total_loss(logits, targets, num_classes=2, pos_weight=None, w_edge=0.2):
    return bce_edge_combo(logits, targets, num_classes, pos_weight, w_edge) + \
           dice_loss_only(logits, targets, num_classes)

def train_one_epoch(model, loader, optimizer, device, scaler, num_classes=2,
                    pos_weight=None, w_edge=0.2, deep_supervision=True):
    model.train()
    meter = BinaryConfusionMeter()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        # 确保 mask 为 [B,1,H,W] 且 {0,1}
        if masks.ndim == 3: masks = masks.unsqueeze(1)
        if masks.dtype != torch.long:
            masks = (masks > 0.5).long() if masks.max() <= 1 else (masks >= 128).long()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            outputs = model(images)
            if deep_supervision and isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
                logits_main, logits_aux2, logits_aux3 = outputs[:3]
                loss = total_loss(
                    logits_main=logits_main,
                    logits_aux2=logits_aux2,
                    logits_aux3=logits_aux3,
                    targets=masks,
                    num_classes=num_classes,
                    w_edge=w_edge,
                    pos_weight=pos_weight,
                )
                logits = logits_main
            else:
                logits = get_main_logits(outputs)
                loss = single_head_total_loss(
                    logits=logits,
                    targets=masks,
                    num_classes=num_classes,
                    pos_weight=pos_weight,
                    w_edge=w_edge,
                )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss)

        preds = torch.argmax(logits, dim=1)
        meter.update(preds, masks)
        pbar.set_postfix(loss=f"{float(loss):.4f}")

    avg_loss = running_loss / max(1, len(loader))
    metrics = meter.summary()
    return avg_loss, metrics

@torch.no_grad()
def validate_one_epoch(model, loader, device, num_classes=2, deep_supervision=True):
    model.eval()
    meter = BinaryConfusionMeter()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Val  ", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        if masks.ndim == 3: masks = masks.unsqueeze(1)
        if masks.dtype != torch.long:
            masks = (masks > 0.5).long() if masks.max() <= 1 else (masks >= 128).long()

        with autocast(enabled=True):
            outputs = model(images)
            if deep_supervision and isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
                logits_main, logits_aux2, logits_aux3 = outputs[:3]
                loss = total_loss(
                    logits_main=logits_main,
                    logits_aux2=logits_aux2,
                    logits_aux3=logits_aux3,
                    targets=masks,
                    num_classes=num_classes,
                )
                logits = logits_main
            else:
                logits = get_main_logits(outputs)
                loss = single_head_total_loss(
                    logits=logits,
                    targets=masks,
                    num_classes=num_classes,
                )

        running_loss += float(loss)

        preds = torch.argmax(logits, dim=1)
        meter.update(preds, masks)
        pbar.set_postfix(loss=f"{float(loss):.4f}")

    avg_loss = running_loss / max(1, len(loader))
    metrics = meter.summary()
    return avg_loss, metrics


def main():
    set_seed(0)
    train_path = 'train_npy/image'
    train_mask_path = 'train_npy/mask'
    val_path = 'val_npy/image'
    val_mask_path = 'val_npy/mask'
    #2 head
    os.makedirs('logs_v2', exist_ok=True)
    os.makedirs('models_v2', exist_ok=True)
    transformer = DualCompose([
        DualResize((512, 512)),
    ])


    train_dataset = ImageSliceDataset(
        img_npy_folder=train_path, mask_folder=train_mask_path,
        step=400, transform=transformer,
        visualize_full=False, visualize_slice=False,
        save_img_folder='demo_train_pickf/image', save_mask_folder='demo_train_for/mask'
    )
    val_dataset = ImageSliceDataset(
        img_npy_folder=val_path, mask_folder=val_mask_path,
        step=400, transform=transformer,
        visualize_full=False, visualize_slice=False,
        save_img_folder='demo_train_pickf/image', save_mask_folder='demo_train_for/mask'
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=1, pin_memory=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        try:
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True

    deep_supervision = True
    model = PickFormer(num_classes=2).to(device)


    base_lr = 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    num_epochs = 100
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # ===== AMP =====
    scaler = GradScaler(enabled=True)


    best_val_loss = float('inf')
    best_val_dice = -1.0
    best_state_loss = None
    best_state_dice = None

    log_file = open('logs_v2/training_log.txt', 'a', encoding='utf-8')

    for epoch in range(1, num_epochs + 1):
      
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, scaler,
            num_classes=2, pos_weight=None, w_edge=0.2,
            deep_supervision=deep_supervision
        )

     
        val_loss, val_metrics = validate_one_epoch(
            model, val_loader, device, num_classes=2,
            deep_supervision=deep_supervision
        )

 
        scheduler.step()

 
        msg = (f"Epoch [{epoch}/{num_epochs}] "
               f"| TrainLoss {train_loss:.4f}  mIoU {train_metrics['mIoU']:.4f}  Dice {train_metrics['Dice']:.4f}  "
               f"Kappa {train_metrics['Kappa']:.4f}  F1 {train_metrics['F1-score']:.4f} || "
               f"ValLoss {val_loss:.4f}  mIoU {val_metrics['mIoU']:.4f}  Dice {val_metrics['Dice']:.4f}  "
               f"Kappa {val_metrics['Kappa']:.4f}  F1 {val_metrics['F1-score']:.4f}")
        print(msg)
        log_file.write(msg + "\n"); log_file.flush()


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_loss = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state_loss, 'models_v2/best_by_val_loss.pth')


        if val_metrics['Dice'] > best_val_dice:
            best_val_dice = val_metrics['Dice']
            best_state_dice = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state_dice, 'models_v2/best_by_val_dice.pth')

     
        if epoch % 2 == 0:
            ckpt_path = os.path.join("models_v2", f"ckpt_epoch_{epoch:04d}.pth")
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                path=ckpt_path
            )

    log_file.close()
    print("Training done.")
    if best_state_loss is not None:
        print(f"Saved: models_v2/best_by_val_loss.pth  (ValLoss={best_val_loss:.4f})")
    if best_state_dice is not None:
        print(f"Saved: models_v2/best_by_val_dice.pth  (ValDice={best_val_dice:.4f})")

if __name__ == "__main__":
    main()

