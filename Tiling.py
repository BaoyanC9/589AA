import cv2
import os
import numpy as np
from pathlib import Path

def tile_image(image, tile_size=640, stride=320):
    """
    将图像切割成小块
    """
    h, w = image.shape[:2]
    tiles = []
    positions = []
    
    # 确保我们至少有一个块
    if h < tile_size:
        h = tile_size
        # 通过填充图像来处理高度不足的情况
        padded = np.zeros((tile_size, w, 3), dtype=image.dtype)
        padded[:image.shape[0], :, :] = image
        image = padded

    if w < tile_size:
        w = tile_size
        # 通过填充图像来处理宽度不足的情况
        padded = np.zeros((h, tile_size, 3), dtype=image.dtype)
        padded[:, :image.shape[1], :] = image
        image = padded

    # 生成切片
    for i in range(0, h - tile_size + 1, stride):
        for j in range(0, w - tile_size + 1, stride):
            tile = image[i:i+tile_size, j:j+tile_size]
            tiles.append(tile)
            positions.append((i, j))

    # 处理边界情况，确保覆盖整张图像的右下角
    if h > tile_size:
        i = h - tile_size
        for j in range(0, w - tile_size + 1, stride):
            tile = image[i:i+tile_size, j:j+tile_size]
            tiles.append(tile)
            positions.append((i, j))

    if w > tile_size:
        j = w - tile_size
        for i in range(0, h - tile_size + 1, stride):
            tile = image[i:i+tile_size, j:j+tile_size]
            tiles.append(tile)
            positions.append((i, j))

    # 添加右下角的块
    if h > tile_size and w > tile_size:
        i, j = h - tile_size, w - tile_size
        tile = image[i:i+tile_size, j:j+tile_size]
        tiles.append(tile)
        positions.append((i, j))

    return tiles, positions

def process_labels_for_tile(label_file, tile_position, tile_size, original_shape):
    """
    为tile调整标签坐标，返回非空标签内容列表
    """
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        tile_lines = []
        tile_y, tile_x = tile_position
        img_h, img_w = original_shape[:2]
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            class_id = parts[0]
            # YOLO格式: class_id x_center y_center width height (normalized)
            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            width = float(parts[3]) * img_w
            height = float(parts[4]) * img_h
            
            # 转换为绝对坐标
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # 检查目标是否在tile内
            tile_x1, tile_y1 = tile_x, tile_y
            tile_x2, tile_y2 = tile_x + tile_size, tile_y + tile_size
            
            # 计算交集
            inter_x1 = max(x1, tile_x1)
            inter_y1 = max(y1, tile_y1)
            inter_x2 = min(x2, tile_x2)
            inter_y2 = min(y2, tile_y2)
            
            # 如果有交集
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                # 计算交集在tile中的相对位置
                new_x_center = (inter_x1 + inter_x2) / 2 - tile_x1
                new_y_center = (inter_y1 + inter_y2) / 2 - tile_y1
                new_width = inter_x2 - inter_x1
                new_height = inter_y2 - inter_y1
                
                # 归一化到tile尺寸
                new_x_center /= tile_size
                new_y_center /= tile_size
                new_width /= tile_size
                new_height /= tile_size
                
                # 确保坐标在有效范围内
                if 0 <= new_x_center <= 1 and 0 <= new_y_center <= 1 and new_width > 0 and new_height > 0:
                    tile_lines.append(f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}\n")
        return tile_lines
    except Exception as e:
        print(f"处理标签文件时出错 {label_file}: {e}")
        return []

def process_dataset_split(input_images_dir, input_labels_dir, output_dir, tile_size=640, stride=320):
    """
    处理数据集的一个分割（train/val/test），只保存有目标的tile
    """
    input_images_path = Path(input_images_dir)
    input_labels_path = Path(input_labels_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    (output_path / 'images').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_files = list(input_images_path.glob("*.jpg")) + list(input_images_path.glob("*.png"))
    
    tile_count = 0
    for img_file in image_files:
        # 读取图像
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"无法读取图像: {img_file}")
            continue
            
        # 切割图像
        tiles, positions = tile_image(image, tile_size, stride)
        
        # 获取对应的标签文件
        label_file = input_labels_path / (img_file.stem + '.txt')
        
        # 处理每个tile
        for idx, (tile, pos) in enumerate(zip(tiles, positions)):
            # 如果存在标签文件，则处理标签
            if label_file.exists():
                tile_labels = process_labels_for_tile(label_file, pos, tile_size, image.shape)
                if len(tile_labels) > 0:
                    # 保存tile图像
                    tile_filename = f"{img_file.stem}_{idx:04d}.jpg"
                    cv2.imwrite(str(output_path / 'images' / tile_filename), tile)
                    # 保存tile的标签文件
                    with open(output_path / 'labels' / f"{img_file.stem}_{idx:04d}.txt", 'w') as f:
                        f.writelines(tile_labels)
                    tile_count += 1
            else:
                # 没有标签文件则不保存tile
                continue

    print(f"处理完成 {input_images_dir}，共生成 {tile_count} 个有效tiles")
    return tile_count

def process_wsodd_dataset():
    """
    处理完整的数据集
    """
    dataset_root = "datasets/Water_splits"
    output_root = "datasets/Water_splits_tiled2"

    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"正在处理 {split} 数据集...")
        input_images_dir = os.path.join(dataset_root, split, 'images')
        input_labels_dir = os.path.join(dataset_root, split, 'labels')
        output_dir = os.path.join(output_root, split)
        process_dataset_split(input_images_dir, input_labels_dir, output_dir)
    print("数据集处理完成！")

if __name__ == "__main__":
    process_wsodd_dataset()