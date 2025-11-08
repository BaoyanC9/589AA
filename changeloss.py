import torch
import torch.nn.functional as F

# ---------------------------
# Basic IoU (xyxy format)
# ---------------------------
def bbox_iou_xyxy(box1, box2, eps=1e-7):
    """
    box1, box2: (..., 4) in xyxy
    returns IoU: (...,)
    """
    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (box1[..., 2] - box1[..., 0]).clamp(min=0) * (box1[..., 3] - box1[..., 1]).clamp(min=0)
    area2 = (box2[..., 2] - box2[..., 0]).clamp(min=0) * (box2[..., 3] - box2[..., 1]).clamp(min=0)

    union = area1 + area2 - inter + eps
    iou = inter / union
    return iou

def iou_loss(box1, box2, reduction='mean'):
    iou = bbox_iou_xyxy(box1, box2)
    loss = 1.0 - iou  # IoU loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

# ---------------------------
# EIoU (Efficient IoU) loss
# Paper-style simplified implementation
# ---------------------------
def eiou_loss(box1, box2, reduction='mean', eps=1e-7):
    """
    EIoU = (1 - IoU) + center_distance_term + wh_term
    All boxes in xyxy, shape (..., 4)
    """
    iou = bbox_iou_xyxy(box1, box2, eps=eps)

    # centers and sizes
    x1c = (box1[..., 0] + box1[..., 2]) / 2
    y1c = (box1[..., 1] + box1[..., 3]) / 2
    x2c = (box2[..., 0] + box2[..., 2]) / 2
    y2c = (box2[..., 1] + box2[..., 3]) / 2

    w1 = (box1[..., 2] - box1[..., 0]).clamp(min=eps)
    h1 = (box1[..., 3] - box1[..., 1]).clamp(min=eps)
    w2 = (box2[..., 2] - box2[..., 0]).clamp(min=eps)
    h2 = (box2[..., 3] - box2[..., 1]).clamp(min=eps)

    # enclosing box (for normalization)
    xc1 = torch.min(box1[...,0], box2[...,0])
    yc1 = torch.min(box1[...,1], box2[...,1])
    xc2 = torch.max(box1[...,2], box2[...,2])
    yc2 = torch.max(box1[...,3], box2[...,3])

    cw = (xc2 - xc1).clamp(min=eps)
    ch = (yc2 - yc1).clamp(min=eps)

    # center distance term (normalized)
    center_dist = ((x1c - x2c)**2 + (y1c - y2c)**2) / (cw**2 + ch**2 + eps)

    # width/height term (normalized to target size, per many EIoU impls)
    w_term = ((w1 - w2)**2) / (w2**2 + eps)
    h_term = ((h1 - h2)**2) / (h2**2 + eps)

    loss = (1.0 - iou) + center_dist + w_term + h_term
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

# ---------------------------
# Focal-EIoU: focal weighting on top of EIoU
# ---------------------------
def focal_eiou_loss(box1, box2, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-7):
    """
    Focal-EIoU = (alpha * (1 - IoU)^gamma) * EIoU
    """
    with torch.no_grad():
        iou = bbox_iou_xyxy(box1, box2, eps=eps).clamp(min=eps, max=1.0)
        focal = alpha * (1.0 - iou)**gamma

    base = eiou_loss(box1, box2, reduction='none', eps=eps)
    loss = focal * base
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

# ---------------------------
# SIoU (Scale-invariant IoU) — 简洁实现
# 注：SIoU有多种实现版本，这里给出常用近似实现
# ---------------------------
def siou_loss(box1, box2, reduction='mean', eps=1e-7):
    """
    A practical SIoU-like loss:
    L = 1 - IoU + angle_term + distance_term + shape_term
    这里采用简化版本：用距离+宽高差作额外惩罚
    """
    iou = bbox_iou_xyxy(box1, box2, eps=eps)

    # centers and sizes
    x1c = (box1[..., 0] + box1[..., 2]) / 2
    y1c = (box1[..., 1] + box1[..., 3]) / 2
    x2c = (box2[..., 0] + box2[..., 2]) / 2
    y2c = (box2[..., 1] + box2[..., 3]) / 2

    w1 = (box1[..., 2] - box1[..., 0]).clamp(min=eps)
    h1 = (box1[..., 3] - box1[..., 1]).clamp(min=eps)
    w2 = (box2[..., 2] - box2[..., 0]).clamp(min=eps)
    h2 = (box2[..., 3] - box2[..., 1]).clamp(min=eps)

    # enclosing box
    xc1 = torch.min(box1[...,0], box2[...,0])
    yc1 = torch.min(box1[...,1], box2[...,1])
    xc2 = torch.max(box1[...,2], box2[...,2])
    yc2 = torch.max(box1[...,3], box2[...,3])
    cw = (xc2 - xc1).clamp(min=eps)
    ch = (yc2 - yc1).clamp(min=eps)

    # distance (normalized)
    distance = ((x1c - x2c)**2 + (y1c - y2c)**2) / (cw**2 + ch**2 + eps)

    # shape term
    shape = ((w1 - w2)**2 / (w2**2 + eps) + (h1 - h2)**2 / (h2**2 + eps)) * 0.5

    loss = (1.0 - iou) + distance + shape
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss
