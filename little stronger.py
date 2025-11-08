import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from ultralytics.utils.loss import BboxLoss
from pathlib import Path
import shutil
# 低光增强 (简单线性调节亮度)
def low_light_augmentation(image):
    return np.clip(image * 0.5, 0, 255)

# 反光增强（随机局部反射）
def reflection_augmentation(image):
    h, w, _ = image.shape
    x1, y1 = np.random.randint(0, w - 100), np.random.randint(0, h - 100)
    x2, y2 = x1 + 100, y1 + 100
    image[y1:y2, x1:x2] = np.clip(image[y1:y2, x1:x2] + 50, 0, 255)
    return image


def process_dataset(src_dir, dst_dir, augmentation_type='low_light'):
    """
    处理数据集应用增强技术
    """
    # 创建目标目录
    os.makedirs(os.path.join(dst_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'val', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'test', 'labels'), exist_ok=True)
    
    # 复制配置文件
    shutil.copy(
        os.path.join(src_dir, 'Water_enhanced_tiled2.yaml'),
        os.path.join(dst_dir, 'Water_enhanced_tiled2_reflection.yaml')
    )
    
    # 处理训练集、验证集和测试集
    for split in ['train', 'val', 'test']:
        print(f"Processing {split} set...")
        images_dir = os.path.join(src_dir, split, 'images')
        labels_dir = os.path.join(src_dir, split, 'labels')
        output_images_dir = os.path.join(dst_dir, split, 'images')
        output_labels_dir = os.path.join(dst_dir, split, 'labels')
        
        # 获取图像文件列表
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 处理每个图像
        for i, image_file in enumerate(image_files):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(image_files)} images")
            
            # 读取图像
            image_path = os.path.join(images_dir, image_file)
            image = cv2.imread(image_path)
            if image is None:
                continue
                
            # 应用增强
            if augmentation_type == 'low_light':
                augmented_image = low_light_augmentation(image)
            elif augmentation_type == 'reflection':
                augmented_image = reflection_augmentation(image)
            else:
                augmented_image = image  # 不应用增强
            
            # 保存增强后的图像
            output_image_path = os.path.join(output_images_dir, image_file)
            cv2.imwrite(output_image_path, augmented_image)
            
            # 复制对应的标签文件
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            output_label_path = os.path.join(output_labels_dir, label_file)
            if os.path.exists(label_path):
                shutil.copy(label_path, output_label_path)
        
        print(f"Finished processing {split} set.")


if __name__ == "__main__":
    # 源数据集和目标数据集路径
    src_dataset_path = "datasets/Water_enhanced_tiled2"
    dst_dataset_path = "datasets/Water_enhanced_tiled2_reflection"

    # 处理数据集应用增强
    print("Processing dataset with reflection augmentation...")
    process_dataset(src_dataset_path, dst_dataset_path, augmentation_type='reflection')

