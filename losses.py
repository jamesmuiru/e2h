"""
losses.py — Loss functions and evaluation metrics for Embed2Heights.

Loss (as specified in architecture):
  Total = 0.45 × (Dice + BCE) + 0.55 × Huber

Metrics (official challenge):
  mIoU_buildings        (25%)
  mIoU_trees            (15%)
  mIoU_water            (15%)
  RMSE_building_height  (25%)
  RMSE_vegetation_height(20%)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════════════════════════════════════

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    p = pred.contiguous().view(-1)
    t = target.contiguous().view(-1)
    inter = (p * t).sum()
    return 1.0 - (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)


def bce_dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # THE FIX: Replace 'NaN' (NoData pixels from TIFFs) with 0.0 before clamping.
    # torch.clamp ignores NaNs, which is what caused the C++ assert to fail!
    p = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    t = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
    
    p = torch.clamp(p, min=1e-6, max=1.0 - 1e-6)
    t = torch.clamp(t, min=0.0, max=1.0)
    
    bce  = F.binary_cross_entropy(p, t, reduction="mean")
    dice = dice_loss(p, t)
    return 0.5 * bce + 0.5 * dice


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    # Protect Huber loss from NaNs as well
    p = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    t = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
    return F.huber_loss(p, t, reduction="mean", delta=delta)


def compute_loss(
    seg_pred:    torch.Tensor,   # (B, 3, H, W)  sigmoid output
    height_pred: torch.Tensor,   # (B, 1, H, W)  softplus output
    label:       torch.Tensor,   # (B, 4, H, W)  [bldg, veg, water, nDSM_log1p]
    w_seg:       float = 0.45,
    w_height:    float = 0.55,
) -> tuple[torch.Tensor, float, float]:
    """
    Returns (total_loss, seg_loss_scalar, height_loss_scalar).
    """
    seg_target    = label[:, :3]    # (B, 3, H, W)
    height_target = label[:, 3:4]   # (B, 1, H, W)

    seg_loss = sum(
        bce_dice_loss(seg_pred[:, c], seg_target[:, c]) for c in range(3)
    ) / 3.0

    h_loss = huber_loss(height_pred, height_target)

    total = w_seg * seg_loss + w_height * h_loss
    
    return total, float(seg_loss.item()), float(h_loss.item())


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ══════════════════════════════════════════════════════════════════════════════

def _iou_binary(pred_prob: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.5) -> float:
    pred  = (pred_prob > threshold).float()
    tgt   = (target    > threshold).float()
    inter = (pred * tgt).sum()
    union = (pred + tgt).clamp(0, 1).sum()
    return float(inter / (union + 1e-6))


def _miou(pred_ch: torch.Tensor, tgt_ch: torch.Tensor) -> float:
    iou_fg = _iou_binary(pred_ch,     tgt_ch,     threshold=0.5)
    iou_bg = _iou_binary(1 - pred_ch, 1 - tgt_ch, threshold=0.5)
    return (iou_fg + iou_bg) / 2.0


def _masked_rmse(pred_m: torch.Tensor, tgt_m: torch.Tensor,
                 mask: torch.Tensor) -> float:
    if mask.sum() == 0:
        return 0.0
    diff2 = (pred_m[mask] - tgt_m[mask]) ** 2
    return float(diff2.mean().sqrt())


def compute_metrics(
    seg_pred:    torch.Tensor,
    height_pred: torch.Tensor,
    label:       torch.Tensor,
) -> dict:
    """
    Compute the 5 official competition metrics for a batch.
    height_pred and label[:,3] are in log1p space — we invert to metres first.
    """
    # Sanitize metric arrays from NaNs before scoring to prevent metric crashes
    seg_pred    = torch.nan_to_num(seg_pred.cpu().detach(), nan=0.0)
    height_pred = torch.nan_to_num(height_pred.cpu().detach(), nan=0.0)
    label       = torch.nan_to_num(label.cpu().detach(), nan=0.0)

    seg_tgt = label[:, :3]       # (B, 3, H, W)
    ht_tgt  = label[:, 3:4]      # (B, 1, H, W)  log1p space

    # Invert log1p → metres
    ht_pred_m = torch.expm1(height_pred).clamp(0)
    ht_tgt_m  = torch.expm1(ht_tgt).clamp(0)

    B = seg_pred.shape[0]

    mIoU_bldg  = np.mean([_miou(seg_pred[b, 0], seg_tgt[b, 0]) for b in range(B)])
    mIoU_tree  = np.mean([_miou(seg_pred[b, 1], seg_tgt[b, 1]) for b in range(B)])
    mIoU_water = np.mean([_miou(seg_pred[b, 2], seg_tgt[b, 2]) for b in range(B)])

    # Height RMSE masked to relevant land-cover pixels
    bldg_mask = seg_tgt[:, 0:1] > 0.5
    veg_mask  = seg_tgt[:, 1:2] > 0.1

    rmse_bldg = _masked_rmse(ht_pred_m, ht_tgt_m, bldg_mask)
    rmse_veg  = _masked_rmse(ht_pred_m, ht_tgt_m, veg_mask)

    return {
        "mIoU_buildings":         float(mIoU_bldg),
        "mIoU_trees":             float(mIoU_tree),
        "mIoU_water":             float(mIoU_water),
        "RMSE_building_height":   rmse_bldg,
        "RMSE_vegetation_height": rmse_veg,
    }


def weighted_score(metrics: dict, weights: dict) -> float:
    """
    Weighted competition score (higher = better).
    IoU metrics contribute directly; RMSE via 1/(1+RMSE) proxy.
    """
    return (
        weights["mIoU_buildings"]         * metrics["mIoU_buildings"] +
        weights["mIoU_trees"]             * metrics["mIoU_trees"] +
        weights["mIoU_water"]             * metrics["mIoU_water"] +
        weights["RMSE_building_height"]   * (1.0 / (1.0 + metrics["RMSE_building_height"])) +
        weights["RMSE_vegetation_height"] * (1.0 / (1.0 + metrics["RMSE_vegetation_height"]))
    )