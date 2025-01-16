import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from utils.transforms import resize, random_transform, mosaic_augment, mixup_augment
import random
from config import cfg

class YOLODataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
            
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                           for path in self.img_files]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.batch_count = 0
        
    def __getitem__(self, index):
        # 加载图像
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path)
        
        # 从标签文件中读取边界框
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        
        boxes = None
        if os.path.exists(label_path):
            boxes = np.loadtxt(label_path).reshape(-1, 5)
        
        # 应用数据增强
        if self.augment:
            # 随机应用Mosaic增强
            if random.random() < cfg.TRAIN.MOSAIC_PROB and len(self.img_files) >= 4:
                # 随机选择另外3张图片
                indices = random.sample(range(len(self.img_files)), 3)
                extra_images = []
                extra_targets = []
                for idx in indices:
                    img_path = self.img_files[idx].rstrip()
                    label_path = self.label_files[idx].rstrip()
                    extra_img = cv2.imread(img_path)
                    if os.path.exists(label_path):
                        extra_boxes = np.loadtxt(label_path).reshape(-1, 5)
                        extra_targets.append(extra_boxes)
                    else:
                        extra_targets.append(np.array([]))
                    extra_images.append(extra_img)
                
                img, boxes = mosaic_augment([img] + extra_images, 
                                          [boxes if boxes is not None else np.array([])] + extra_targets)
            
            # 随机应用MixUp增强
            elif random.random() < cfg.TRAIN.MIXUP_PROB and len(self.img_files) >= 2:
                idx = random.randint(0, len(self.img_files) - 1)
                mix_img_path = self.img_files[idx].rstrip()
                mix_label_path = self.label_files[idx].rstrip()
                mix_img = cv2.imread(mix_img_path)
                if os.path.exists(mix_label_path):
                    mix_boxes = np.loadtxt(mix_label_path).reshape(-1, 5)
                else:
                    mix_boxes = np.array([])
                
                img, boxes = mixup_augment(img, mix_img, 
                                         boxes if boxes is not None else np.array([]), 
                                         mix_boxes)
            
            # 应用基本的数据增强
            img, boxes = random_transform(img, boxes)
            
        # 调整图像大小
        img, boxes = resize(img, boxes, self.img_size)
        
        # 转换为张量
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        
        if boxes is not None and len(boxes) > 0:
            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = torch.from_numpy(boxes)
        else:
            targets = torch.zeros((0, 6))
            
        return img_path, img, targets
    
    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # 移除空目标
        targets = [boxes for boxes in targets if boxes is not None]
        # 添加样本索引到目标
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # 选择新的图像大小
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(320, 608 + 1, 32))
        # 调整图像大小
        imgs = torch.stack([resize_image(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets
    
    def __len__(self):
        return len(self.img_files)