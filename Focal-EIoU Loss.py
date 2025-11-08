# Focal-EIoU 损失函数
def focal_eiou_loss(pred_box, target_box, alpha=0.25, gamma=2.0):
    IoU = IoU_loss(pred_box, target_box)  # 使用标准 IoU 损失
    focal_loss = alpha * (1 - IoU)**gamma
    return focal_loss
# Focal-EIoU 损失函数
def focal_eiou_loss(pred_box, target_box, alpha=0.25, gamma=2.0):
    IoU = IoU_loss(pred_box, target_box)  # 使用标准 IoU 损失
    focal_loss = alpha * (1 - IoU)**gamma
    return focal_loss
