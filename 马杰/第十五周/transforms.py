import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import numpy as np

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            if "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"]
                boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([width, 0, width, 0])
                target["boxes"] = boxes
            if "masks" in target and len(target["masks"]) > 0:
                target["masks"] = target["masks"].flip(-1)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        if isinstance(image, torch.Tensor):
            return image, target
        image = F.to_tensor(image)
        return image, target

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class RandomResize:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size
        
    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        
        # 计算缩放比例
        h, w = image.shape[-2:]
        scale = size / min(h, w)
        
        # 限制最大尺寸
        if h * scale > self.max_size:
            scale = self.max_size / h
        if w * scale > self.max_size:
            scale = self.max_size / w
            
        # 调整图像大小
        new_h = int(h * scale)
        new_w = int(w * scale)
        image = F.resize(image, (new_h, new_w))
        
        # 调整标注
        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * torch.as_tensor([scale, scale, scale, scale])
            target["boxes"] = scaled_boxes
        
        # 调整masks
        if "masks" in target and len(target["masks"]) > 0:
            masks = target["masks"]
            masks = F.interpolate(
                masks.unsqueeze(1).float(),
                size=(new_h, new_w),
                mode='nearest'
            ).squeeze(1).bool()
            target["masks"] = masks
        return image, target