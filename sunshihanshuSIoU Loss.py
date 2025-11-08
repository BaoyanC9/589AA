# SIoU 损失函数
import torch
import torch.nn.functional as F


def SIoU_loss(pred_box, target_box):
    # 计算两个框的交集面积和并集面积
    inter_area = (torch.min(pred_box[..., 2], target_box[..., 2]) - torch.max(pred_box[..., 0], target_box[..., 0])) * \
                 (torch.min(pred_box[..., 3], target_box[..., 3]) - torch.max(pred_box[..., 1], target_box[..., 1]))
    inter_area = F.relu(inter_area)
    pred_area = (pred_box[..., 2] - pred_box[..., 0]) * (pred_box[..., 3] - pred_box[..., 1])
    target_area = (target_box[..., 2] - target_box[..., 0]) * (target_box[..., 3] - target_box[..., 1])

    # 计算距离
    center_pred = (pred_box[..., 2] + pred_box[..., 0]) / 2, (pred_box[..., 3] + pred_box[..., 1]) / 2
    center_target = (target_box[..., 2] + target_box[..., 0]) / 2, (target_box[..., 3] + target_box[..., 1]) / 2
    distance = torch.sqrt((center_pred[0] - center_target[0]) ** 2 + (center_pred[1] - center_target[1]) ** 2)

    # SIoU 损失
    union_area = pred_area + target_area - inter_area
    loss = (inter_area / union_area) - torch.exp(-distance / (pred_area + target_area + 1e-7))
    return torch.mean(loss)
