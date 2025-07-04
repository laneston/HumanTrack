import os
import cv2
import shutil
import scipy.io
import numpy as np
from tqdm import tqdm

# 配置路径
mat_path = 'mpii/mpii_human_pose_v1_u12_1.mat'
image_dir = 'mpii/images'
output_train_dir = 'mpii/train'
output_val_dir = 'mpii/val'

# 创建输出目录
os.makedirs(os.path.join(output_train_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(output_train_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_val_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(output_val_dir, 'images'), exist_ok=True)

# 加载MATLAB文件
data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
annolist = data['RELEASE'].annolist

# 获取训练集和验证集索引
train_indices = np.where(data['RELEASE'].img_train == 1)[0]  # 训练集索引
val_indices = np.where(data['RELEASE'].img_train == 0)[0]    # 验证集索引

# MPII关键点顺序 (16个点)
mpii_kpts_order = [
    'rank', 'rkne', 'rhip',  # 0,1,2
    'lhip', 'lkne', 'lank',  # 3,4,5
    'pelv', 'thrx', 'neck',  # 6,7,8
    'head', 'rwri', 'relb',  # 9,10,11
    'rsho', 'lsho', 'lelb',  # 12,13,14
    'lwri'                  # 15
]

def process_annotation(anno, img_width, img_height):
    lines = []
    if not hasattr(anno, 'annorect'):
        return lines

    annorects = anno.annorect if isinstance(anno.annorect, np.ndarray) else [anno.annorect]
    
    for rect in annorects:
        if not hasattr(rect, 'annopoints') or not hasattr(rect.annopoints, 'point'):
            continue
            
        # 初始化关键点 (x, y, visibility)
        kpts = np.zeros((16, 3))  # [x, y, v] 默认v=0 (未标注)
        
        # 解析关键点
        points = rect.annopoints.point
        points = points if isinstance(points, np.ndarray) else [points]
        for point in points:
            kpt_id = point.id
            x, y = point.x, point.y
            vis = 1 if (hasattr(point, 'is_visible') and point.is_visible == 1) else 0
            kpts[kpt_id] = [x, y, 2 if vis == 1 else 1]  # Ultralytics: 2=可见, 1=遮挡
        
        # 计算边界框 (包含所有关键点的最小矩形)
        valid_kpts = kpts[kpts[:, 2] > 0]  # 只考虑标注的点
        if len(valid_kpts) == 0:
            continue
        
        x_min, y_min = np.min(valid_kpts[:, :2], axis=0)
        x_max, y_max = np.max(valid_kpts[:, :2], axis=0)
        
        # 扩展边界框 (10%)
        width = x_max - x_min
        height = y_max - y_min
        x_min = max(0, x_min - width * 0.05)
        y_min = max(0, y_min - height * 0.05)
        x_max = min(img_width, x_max + width * 0.05)
        y_max = min(img_height, y_max + height * 0.05)
        
        # 归一化坐标
        bbox_center_x = (x_min + x_max) / 2 / img_width
        bbox_center_y = (y_min + y_max) / 2 / img_height
        bbox_width = (x_max - x_min) / img_width
        bbox_height = (y_max - y_min) / img_height
        
        # 归一化关键点
        normalized_kpts = kpts.copy()
        normalized_kpts[:, 0] /= img_width
        normalized_kpts[:, 1] /= img_height
        
        # 格式: class bbox_x bbox_y bbox_w bbox_h kpt1_x kpt1_y kpt1_v ... kpt16_x kpt16_y kpt16_v
        line = [0, bbox_center_x, bbox_center_y, bbox_width, bbox_height]
        for kpt in normalized_kpts:
            line.extend(kpt)
        lines.append(line)
    
    return lines

def process_dataset(indices, output_dir, set_name):
    """处理数据集子集（训练集或验证集）"""
    print(f"Processing {set_name} set with {len(indices)} images...")
    
    # 创建输出目录
    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # 进度条设置
    progress_bar = tqdm(indices, desc=set_name)
    
    for idx in progress_bar:
        anno = annolist[idx]
        image_name = anno.image.name
        img_path = os.path.join(image_dir, image_name)
        
        # 更新进度条描述
        progress_bar.set_postfix({"Image": image_name[:20] + '...' if len(image_name) > 20 else image_name})
        
        if not os.path.exists(img_path):
            print(f"\nWarning: Image not found - {img_path}")
            continue
        
        try:
            # 复制图像到目标目录
            dst_img_path = os.path.join(output_images_dir, image_name)
            if not os.path.exists(dst_img_path):
                shutil.copy2(img_path, dst_img_path)
            
            # 读取图像尺寸
            img = cv2.imread(img_path)
            if img is None:
                raise IOError(f"Failed to read image: {img_path}")
                
            img_height, img_width = img.shape[:2]
            
            # 处理标注
            lines = process_annotation(anno, img_width, img_height)
            
            # 写入标签文件
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(output_labels_dir, label_name)
            with open(label_path, 'w') as f:
                for line in lines:
                    # 将浮点数格式化为6位小数，整数保持不变
                    line_str = ' '.join([f'{x:.6f}' if isinstance(x, float) else str(int(x)) for x in line])
                    f.write(line_str + '\n')
        
        except Exception as e:
            print(f"\nError processing {image_name}: {str(e)}")
            continue

# 处理训练集和验证集
process_dataset(train_indices, output_train_dir, "Training")
process_dataset(val_indices, output_val_dir, "Validation")

# 创建数据集配置文件
def create_yaml_config():
    yaml_content = f"""# MPII dataset configuration for YOLOv8 Pose
path: {os.path.abspath('mpii')}  # dataset root dir
train: {os.path.abspath(output_train_dir)}/images  # train images
val: {os.path.abspath(output_val_dir)}/images  # val images

# Keypoints
kpt_shape: [16, 3]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visibility)
flip_idx: [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]  # keypoint flip index (left-right swaps)

# Classes
names:
  0: person
"""

    config_path = 'mpii-pose.yaml'
    with open(config_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nDataset configuration file created: {os.path.abspath(config_path)}")

# 创建YAML配置文件
create_yaml_config()

# 数据集统计
print("\nDataset conversion completed successfully!")
print(f"Training set size: {len(train_indices)} images")
print(f"Validation set size: {len(val_indices)} images")
print(f"Output directories:")
print(f"- Training: {os.path.abspath(output_train_dir)}")
print(f"- Validation: {os.path.abspath(output_val_dir)}")
print(f"Total images processed: {len(train_indices) + len(val_indices)}")