# test_like_train_val.py
import os
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.utils as vutils
import torch.nn.functional as F

from util_new import ImageSliceDataset, DualCompose, DualResize
from PickFormer_v2 import PickFormer 

class BinaryConfusionMeter:
    def __init__(self):
        self.tp = 0; self.fp = 0; self.fn = 0; self.tn = 0

    @torch.no_grad()
    def update(self, preds: torch.Tensor, masks: torch.Tensor):
        # [B,1,H,W] / [B,H,W]
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
        dice_bg = 2*tn / (2*tn + fn + eps)
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

def get_main_logits(outputs):
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def to_uint8(x: torch.Tensor):
    # x: [H,W] 或 [1,H,W]
    if x.ndim == 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    x = x.float()

    minv = torch.amin(x)
    maxv = torch.amax(x)
    if (maxv - minv) > 1e-12:
        x = (x - minv) / (maxv - minv)
    else:
        x = torch.zeros_like(x)
    x = (x * 255.0).clamp(0, 255).byte()
    return x

def save_mask_png(mask01: torch.Tensor, path: str):
    # mask01: [H,W] (0/1) -> 0/255
    png = (mask01.byte() * 255)
    vutils.save_image(png.unsqueeze(0).float() / 255.0, path)

def save_prob_png(prob01: torch.Tensor, path: str):
    # prob01: [H,W] in [0,1] -> 0/255 灰度
    png = (prob01.clamp(0,1) * 255.0).byte()
    vutils.save_image(png.unsqueeze(0).float() / 255.0, path)

def make_overlay_gray_with_prob(img: torch.Tensor, prob: torch.Tensor):
    g = to_uint8(img)          # [H,W] 0~255
    p = (prob.clamp(0,1)*255).byte()
    H, W = g.shape

    base = torch.stack([g, g, g], dim=0).float()  # [3,H,W]
    base[0] = (base[0] + p.float()).clamp(0, 255)
    return (base / 255.0)


@torch.no_grad()
def eval_like_train_validate(model, loader, device, out_dirs, num_classes=2, deep_supervision=True):
    """
    out_dirs: dict with keys ['pred','prob','overlay','gt','input']
    """
    os.makedirs(out_dirs.get('pred','demo_test_pred/pred'), exist_ok=True)
    os.makedirs(out_dirs.get('prob','demo_test_pred/prob'), exist_ok=True)
    os.makedirs(out_dirs.get('overlay','demo_test_pred/overlay'), exist_ok=True)
    os.makedirs(out_dirs.get('gt','demo_test_pred/gt'), exist_ok=True)
    os.makedirs(out_dirs.get('input','demo_test_pred/input'), exist_ok=True)

    model.eval()
    meter = BinaryConfusionMeter()
    pbar = tqdm(loader, desc="Eval(like-validate)", leave=False)

    global_idx = 0
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)   # [B,1,512,512] 
        masks  = masks.to(device, non_blocking=True)    # [B,1,512,512] or [B,512,512]


        if masks.ndim == 3:  # [B,H,W] -> [B,1,H,W]
            masks = masks.unsqueeze(1)
        if masks.dtype != torch.long:
            masks = (masks > 0.5).long() if masks.max() <= 1 else (masks >= 128).long()

        logits = get_main_logits(model(images))         # [B,2,H,W]
        probs  = F.softmax(logits, dim=1)               # [B,2,H,W]
        prob_fg = probs[:, 1, ...]                      # [B,H,W]
        preds  = torch.argmax(logits, dim=1)            # [B,H,W]

        meter.update(preds, masks)

        B = images.size(0)
        for b in range(B):
            fname = f"idx_{global_idx:06d}.png"
            if out_dirs.get('input'):
                img_gray = to_uint8(images[b, 0].detach().cpu())   # [H,W]
                vutils.save_image(img_gray.float().unsqueeze(0)/255.0,
                                   os.path.join(out_dirs['input'], fname))
            # GT
            if out_dirs.get('gt'):
                gt01 = (masks[b, 0].detach().cpu() > 0).to(torch.uint8)  # [H,W]
                save_mask_png(gt01, os.path.join(out_dirs['gt'], fname))
     
            if out_dirs.get('prob'):
                prob = prob_fg[b].detach().cpu()  # [H,W] 0~1
                save_prob_png(prob, os.path.join(out_dirs['prob'], fname))
      
            if out_dirs.get('pred'):
                pred01 = (preds[b].detach().cpu() > 0).to(torch.uint8)
                save_mask_png(pred01, os.path.join(out_dirs['pred'], fname))
        
            if out_dirs.get('overlay'):
                img_overlay = make_overlay_gray_with_prob(images[b, 0].detach().cpu(),
                                                         prob_fg[b].detach().cpu())  # [3,H,W] 0~1
                vutils.save_image(img_overlay, os.path.join(out_dirs['overlay'], fname))

            global_idx += 1

    return meter.summary()


def main():
    val_path = 'test_npy/image'
    val_mask_path = 'test_npy/mask'
    batch_size = 8
    num_workers = 1
    step = 400
    deep_supervision = True  


    transformer = DualCompose([DualResize((512, 512))])

    val_dataset = ImageSliceDataset(
        img_npy_folder=val_path, mask_folder=val_mask_path,
        step=step, transform=transformer,
        visualize_full=False, visualize_slice=False,
        save_img_folder='demo_train_pickf_2/image', save_mask_folder='demo_train_for_2/mask'
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Device: {device}")

    model = PickFormer(num_classes=2).to(device)

    weights_path = 'models_v2/best_by_val_loss.pth'
    state = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state, strict=True)
    print(f"[Info] Loaded weights: {weights_path}")


    out_dirs = {
        'pred':    'demo_test_pred_occ/pred',
        'prob':    'demo_test_pred_occ/prob',
        'overlay': 'demo_test_pred_occ/overlay',
        'gt':      'demo_test_pred_occ/gt',
        'input':   'demo_test_pred_occ/input',
    }

    metrics = eval_like_train_validate(model, val_loader, device, out_dirs,
                                       num_classes=2, deep_supervision=deep_supervision)

    print("\n== Eval (aligned with training validate_one_epoch) ==")
    print(json.dumps(metrics, indent=2))

    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs('eval_logs_occ', exist_ok=True)
    with open(f'eval_logs_occ/val_like_train_{ts}.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved metrics to eval_logs/val_like_train_{ts}.json")
    print(f"[OK] Images saved under: {os.path.abspath('demo_test_pred')}")

if __name__ == "__main__":
    main()


