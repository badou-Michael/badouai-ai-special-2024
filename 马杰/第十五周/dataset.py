import torch
from torch.utils.data import Dataset
import numpy as np
from pycocotools.coco import COCO
import cv2
import os
from .transforms import ToTensor
from torch.nn import functional as F
from .box_utils import box_transform

class CocoDataset(Dataset):
    def __init__(self, image_dir, annot_path, config, transform=None):
        self.image_dir = image_dir
        self.coco = COCO(annot_path)
        self.config = config
        self.transform = transform if transform is not None else ToTensor()
        
        # 获取所有图片ID
        self.image_ids = list(self.coco.imgs.keys())
        
        self.proposals = None  # 需要在适当的时候初始化
        
    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        # 加载图片
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        image = cv2.imread(os.path.join(self.image_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载标注
        annot_ids = self.coco.getAnnIds(imgIds=image_id)
        annots = self.coco.loadAnns(annot_ids)
        
        # 处理标注数据
        boxes = []
        masks = []
        class_ids = []
        
        for annot in annots:
            # 跳过无效标注
            if annot['bbox'][2] < 1 or annot['bbox'][3] < 1:
                continue
                
            # 边界框
            x, y, w, h = annot['bbox']
            boxes.append([x, y, x + w, y + h])
            
            # 掩码
            mask = self.coco.annToMask(annot)
            masks.append(mask)
            
            # 类别ID
            class_ids.append(annot['category_id'])
        
        # 转换为numpy数组
        boxes = np.array(boxes, dtype=np.float32)
        masks = np.array(masks, dtype=np.uint8)
        class_ids = np.array(class_ids, dtype=np.int64)
        
        # 图像预处理
        image, window, scale, padding = self.resize_image(image)
        
        # 调整boxes和masks
        boxes = self.adjust_boxes(boxes, scale, padding)
        masks = self.adjust_masks(masks, padding)
        
        # 转换为tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        boxes = torch.from_numpy(boxes)
        masks = torch.from_numpy(masks)
        class_ids = torch.from_numpy(class_ids)
        
        if len(boxes) == 0:
            # 处理无标注的情况
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM), dtype=torch.uint8)
            class_ids = torch.zeros(0, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'masks': masks,
            'class_ids': class_ids,
            'image_id': image_id
        }
        
        # 添加RPN训练目标
        target.update({
            'rpn_labels': self.generate_rpn_labels(boxes),
            'rpn_bbox_targets': self.generate_rpn_bbox_targets(boxes),
            'box_targets': self.generate_box_targets(boxes, class_ids),
            'mask_targets': self.generate_mask_targets(masks, boxes)
        })
        
        if self.transform is not None:
            image, target = self.transform(image, target)
        
        return image, target
        
    def resize_image(self, image):
        """调整图像大小"""
        # 计算缩放比例
        h, w = image.shape[:2]
        scale = min(
            self.config.IMAGE_MAX_DIM / max(h, w),
            self.config.IMAGE_MIN_DIM / min(h, w)
        )
        
        # 计算新的尺寸
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # 初始化padding变量
        pad_h = 0
        pad_w = 0
        
        # 调整图像大小
        image = cv2.resize(image, (new_w, new_h))
        
        # 填充到指定大小
        if self.config.IMAGE_PADDING:
            pad_h = self.config.IMAGE_MAX_DIM - new_h
            pad_w = self.config.IMAGE_MAX_DIM - new_w
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)))
            
        return image, (0, 0, new_h, new_w), scale, (pad_h, pad_w)
        
    def adjust_boxes(self, boxes, scale, padding):
        """调整边界框"""
        boxes = boxes * scale
        if self.config.IMAGE_PADDING:
            pad_h, pad_w = padding
            boxes[:, [0, 2]] += pad_w // 2
            boxes[:, [1, 3]] += pad_h // 2
        return boxes
        
    def adjust_masks(self, masks, padding):
        """调整掩码"""
        if self.config.IMAGE_PADDING:
            pad_h, pad_w = padding
            masks = np.pad(masks, ((0, 0), (0, pad_h), (0, pad_w)))
        return masks 

    def generate_rpn_labels(self, boxes):
        """生成RPN分类标签"""
        if self.proposals is None:
            # 需要先初始化proposals
            feature_shapes = [(self.config.IMAGE_SHAPE[0]//stride, 
                              self.config.IMAGE_SHAPE[1]//stride)
                             for stride in (4, 8, 16, 32, 64)]
            self.proposals = [generate_anchors(
                self.config.ANCHOR_SCALES,
                self.config.ANCHOR_RATIOS,
                shape,
                stride,
                self.config.ANCHOR_STRIDE
            ) for shape, stride in zip(feature_shapes, (4, 8, 16, 32, 64))]
        
        # 生成anchors
        anchors = self.generate_anchors()
        # 计算IoU
        ious = box_iou(anchors, boxes)
        # 分配标签
        labels = torch.zeros(len(anchors), dtype=torch.long)
        max_ious, max_idx = ious.max(dim=1)
        # 正样本
        pos_mask = max_ious >= self.config.ROI_POSITIVE_THRESHOLD
        labels[pos_mask] = 1
        # 负样本
        neg_mask = max_ious < self.config.ROI_NEGATIVE_THRESHOLD
        labels[neg_mask] = 0
        # 忽略的样本
        labels[~(pos_mask | neg_mask)] = -1
        return labels

    def generate_rpn_bbox_targets(self, boxes):
        """生成RPN回归目标"""
        anchors = self.generate_anchors()
        ious = box_iou(anchors, boxes)
        max_ious, max_idx = ious.max(dim=1)
        # 只为正样本生成回归目标
        pos_mask = max_ious >= self.config.ROI_POSITIVE_THRESHOLD
        matched_boxes = boxes[max_idx[pos_mask]]
        pos_anchors = anchors[pos_mask]
        # 计算回归目标
        targets = self.box_transform(pos_anchors, matched_boxes)
        return targets

    def generate_box_targets(self, boxes, class_ids):
        """生成检测头回归目标"""
        # 类似RPN的回归目标生成
        return self.box_transform(self.proposals, boxes)

    def generate_mask_targets(self, masks, boxes):
        """生成mask目标"""
        # 将mask裁剪到proposal区域并resize到固定大小
        mask_targets = []
        for mask, box in zip(masks, boxes):
            x1, y1, x2, y2 = box.int()
            mask_target = mask[y1:y2, x1:x2]
            mask_target = F.interpolate(
                mask_target.unsqueeze(0).unsqueeze(0).float(),
                size=(28, 28),
                mode='bilinear',
                align_corners=False
            ).squeeze().bool()
            mask_targets.append(mask_target)
        return torch.stack(mask_targets)

    def generate_anchors(self):
        """生成所有特征层的anchors"""
        if self.proposals is None:
            self._init_proposals()
        return torch.cat(self.proposals, dim=0)

    def _init_proposals(self):
        """初始化proposals"""
        from .anchors import generate_anchors
        feature_shapes = [(self.config.IMAGE_SHAPE[0]//stride, 
                          self.config.IMAGE_SHAPE[1]//stride)
                         for stride in (4, 8, 16, 32, 64)]
        self.proposals = [generate_anchors(
            self.config.ANCHOR_SCALES,
            self.config.ANCHOR_RATIOS,
            shape,
            stride,
            self.config.ANCHOR_STRIDE
        ) for shape, stride in zip(feature_shapes, (4, 8, 16, 32, 64))]