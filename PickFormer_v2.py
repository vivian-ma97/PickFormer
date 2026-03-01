# pickformer_smooth.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import timm
from timm.models.layers import DropPath

# ===============================
# 频率通道注意力（保留你原思路）
# ===============================
class FrequencyChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16,
                 low_ratio: float = 0.10, high_ratio: float = 0.40):
        super().__init__()
        self.channels = channels
        self.reduction = max(1, reduction)
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio
        self.freq_mix = nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=True)
        self.mlp = nn.Sequential(
            nn.Linear(channels, max(1, channels // self.reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // self.reduction), channels, bias=False),
            nn.Sigmoid()
        )

    @staticmethod
    def _fftshift2d(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return torch.roll(torch.roll(x, h // 2, dims=-2), w // 2, dims=-1)

    def _make_masks(self, h, w, device):
        yy, xx = torch.meshgrid(torch.arange(h, device=device),
                                torch.arange(w, device=device), indexing='ij')
        cy, cx = h // 2, w // 2
        r = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)
        rmax = float(min(h, w)) / 2.0 + 1e-6
        low = (r <= self.low_ratio * rmax).float()
        high = (r >= self.high_ratio * rmax).float()
        return low[None, None], high[None, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        spatial_desc = x.mean(dim=(2, 3))  # [B, C]
        spec = torch.fft.fft2(x.float(), norm="ortho")
        mag = torch.abs(spec)
        mag = self._fftshift2d(mag)
        low_mask, high_mask = self._make_masks(H, W, x.device)
        eps = 1e-6
        low_energy = (mag * low_mask).sum((2, 3)) / (low_mask.sum() + eps)
        high_energy = (mag * high_mask).sum((2, 3)) / (high_mask.sum() + eps)
        freq_weight = torch.softmax(self.freq_mix, dim=0)
        freq_desc = freq_weight[0] * low_energy + freq_weight[1] * high_energy
        gate = self.mlp(spatial_desc + freq_desc).view(B, C, 1, 1)
        return x * gate

# ===============================
# 基础卷积
# ===============================
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, d=1, s=1, bias=False):
        pad = ((s - 1) + d * (k - 1)) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, padding=pad, dilation=d, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True)
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, d=1, s=1, bias=False):
        pad = ((s - 1) + d * (k - 1)) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, padding=pad, dilation=d, bias=bias),
            nn.BatchNorm2d(out_ch)
        )

class Conv(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, d=1, s=1, bias=False):
        pad = ((s - 1) + d * (k - 1)) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, padding=pad, dilation=d, bias=bias)
        )

# ===============================
# Global-Local Attention + Block（沿用你的）
# ===============================

###change tou
class GlobalLocalAttention(nn.Module):
    def __init__(self, dim=256, num_heads=8, window_size=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv  = Conv(dim, 3 * dim, k=1, bias=qkv_bias)
        self.local = ConvBN(dim, dim, k=3)
        self.proj  = ConvBN(dim, dim, k=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1).reshape(B, C, H, W)
        return self.proj(out + self.local(x))

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)

class Block(nn.Module):
    def __init__(self, dim=128, num_heads=2, mlp_ratio=4., drop_path=0., window_size=8):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = GlobalLocalAttention(dim, num_heads, window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ===============================
# 轻量 SPP（可选，顶部补感受野）
# ===============================
class LiteSPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b1 = nn.Sequential(nn.AdaptiveAvgPool2d(1),  nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.AdaptiveAvgPool2d(2),  nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.AdaptiveAvgPool2d(3),  nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.ReLU(inplace=True))
        self.proj = ConvBNReLU(in_ch + 3*out_ch, out_ch, k=1)

    def forward(self, x):
        H, W = x.shape[-2:]
        y1 = F.interpolate(self.b1(x), size=(H, W), mode='bilinear', align_corners=False)
        y2 = F.interpolate(self.b2(x), size=(H, W), mode='bilinear', align_corners=False)
        y3 = F.interpolate(self.b3(x), size=(H, W), mode='bilinear', align_corners=False)
        return self.proj(torch.cat([x, y1, y2, y3], dim=1))

# ===============================
# 解码器：FPN式逐级融合 + FCA（更平滑）
# ===============================
class Decoder(nn.Module):
    def __init__(self, enc_chs, decode_ch=16, num_classes=2, window_size=8, use_spp=True):
        super().__init__()
        c1, c2, c3, c4 = enc_chs
        self.top = LiteSPP(c4, decode_ch) if use_spp else nn.Conv2d(c4, decode_ch, 1, bias=False)

        # lateral 统一到 decode_ch
        self.lat3 = nn.Conv2d(c3, decode_ch, 1, bias=False)
        self.lat2 = nn.Conv2d(c2, decode_ch, 1, bias=False)

        # 融合后：Block + FCA 细化
        self.p4_blk = Block(decode_ch, num_heads=1, window_size=window_size)
        self.p4_fca = FrequencyChannelAttention(decode_ch)

        self.p3_blk = Block(decode_ch, num_heads=1, window_size=window_size)
        self.p3_fca = FrequencyChannelAttention(decode_ch)

        self.p2_blk = Block(decode_ch, num_heads=1, window_size=window_size)
        self.p2_fca = FrequencyChannelAttention(decode_ch)

        # 最后再平滑几下
        self.smooth3 = ConvBNReLU(decode_ch, decode_ch, k=3)
        self.smooth2 = ConvBNReLU(decode_ch, decode_ch, k=3)

        # segmentation head
        self.head = nn.Sequential(
            ConvBNReLU(decode_ch, decode_ch, k=3),
            nn.Dropout2d(0.1, inplace=True),
            nn.Conv2d(decode_ch, num_classes, 1, bias=True)
        )

    def forward(self, r1, r2, r3, r4, H, W):
        # 顶层
        p4 = self.top(r4)                 # [B,dc,h4,w4]
        p4 = self.p4_fca(self.p4_blk(p4))

        # 融合到 P3
        p3 = self.lat3(r3) + F.interpolate(p4, size=r3.shape[-2:], mode='bilinear', align_corners=False)
        p3 = self.p3_fca(self.p3_blk(p3))
        p3 = self.smooth3(p3)

        # 融合到 P2（再平滑）
        p2 = self.lat2(r2) + F.interpolate(p3, size=r2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.p2_fca(self.p2_blk(p2))
        p2 = self.smooth2(p2)

        # 输出到原尺寸
        logits = self.head(F.interpolate(p2, size=(H, W), mode='bilinear', align_corners=False))
        return logits

# ===============================
# 主网络
# ===============================
class PickFormer(nn.Module):
    def __init__(self, backbone_name="swsl_resnet18", decode_channels=64, num_classes=2, pretrained=True, use_spp=True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            backbone_name, features_only=True, out_indices=(1, 2, 3, 4), pretrained=pretrained
        )
        enc_chs = self.backbone.feature_info.channels()
        self.decoder = Decoder(enc_chs, decode_ch=decode_channels, num_classes=num_classes, use_spp=use_spp)

    def forward(self, x):
        H, W = x.shape[-2:]
        r1, r2, r3, r4 = self.backbone(x)
        return self.decoder(r1, r2, r3, r4, H, W)

# ===============================
# 训练用损失：BCE + Dice + Edge (+ 可选TV)
# ===============================
def _fg_logits(logits, num_classes: int):
    return logits if logits.shape[1] == 1 or num_classes == 1 else logits[:, 1:2, ...]

def dice_loss(logits, targets, eps=1e-6, num_classes=2):
    L = _fg_logits(logits, num_classes)
    probs = torch.sigmoid(L)
    targets = targets.float()
    num = 2 * (probs * targets).sum(dim=(2, 3)) + eps
    den = (probs.pow(2) + targets.pow(2)).sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()

def bce_loss(logits, targets, num_classes=2, pos_weight=None):
    L = _fg_logits(logits, num_classes)
    return F.binary_cross_entropy_with_logits(L, targets.float(), pos_weight=pos_weight)

def edge_loss(logits, targets, num_classes=2):
    L = _fg_logits(logits, num_classes)
    probs = torch.sigmoid(L)
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=logits.device).view(1,1,3,3)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=logits.device).view(1,1,3,3)
    gx_p = F.conv2d(probs, kx, padding=1) + F.conv2d(probs, ky, padding=1)
    gx_t = F.conv2d(targets.float(), kx, padding=1) + F.conv2d(targets.float(), ky, padding=1)
    return F.l1_loss(gx_p, gx_t)

def tv_loss(logits, num_classes=2, weight=1e-4):
    L = _fg_logits(logits, num_classes)
    probs = torch.sigmoid(L)
    dx = probs[:, :, :, 1:] - probs[:, :, :, :-1]
    dy = probs[:, :, 1:, :] - probs[:, :, :-1, :]
    return weight * (dx.abs().mean() + dy.abs().mean())

def total_loss(logits, targets, num_classes=2, pos_weight=None, w_edge=0.2, w_tv=0.0):
    return bce_loss(logits, targets, num_classes, pos_weight) + \
           dice_loss(logits, targets, num_classes) + \
           w_edge * edge_loss(logits, targets, num_classes) + \
           (tv_loss(logits, num_classes) if w_tv > 0 else 0.0)

# ===============================
# 推理后处理：平滑/去小连通域（提升连贯性）
# ===============================
def postprocess_smooth(prob_map: np.ndarray,
                       thresh: float = 0.5,
                       min_cc_area: int = 50,
                       morph_ksize: int = 3) -> np.ndarray:
    """
    prob_map: float HxW in [0,1]
    返回: uint8 HxW {0,255}
    """
    pred = (prob_map >= thresh).astype(np.uint8)
    if morph_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
        # 先闭运算填小洞，再开运算去毛刺
        pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, k)
        pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN,  k)
    # 去小连通域
    if min_cc_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
        keep = np.zeros_like(pred)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_cc_area:
                keep[labels == i] = 1
        pred = keep
    return (pred * 255).astype(np.uint8)

# ===============================
# 快速自检
# ===============================

"""

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 512, 512).to(device)
    model = PickFormer(backbone_name="swsl_resnet18", decode_channels=64, num_classes=2, pretrained=False).to(device)
    y = model(x)  # [B,2,H,W]
    print("Logits:", y.shape)

    # 假 targets
    t = (torch.rand(2, 1, 512, 512) > 0.8).to(device).long()
    loss = total_loss(y, t, num_classes=2, w_edge=0.2, w_tv=1e-4)
    print("Loss:", float(loss))

    # 后处理示例
    with torch.no_grad():
        probs = torch.sigmoid(y[:, 1:2]).cpu().numpy()[0, 0]
        pp = postprocess_smooth(probs, thresh=0.5, min_cc_area=50, morph_ksize=3)
        print("Postprocessed mask shape:", pp.shape, "unique:", np.unique(pp))
"""