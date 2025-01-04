import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import nms, box_iou, roi_align
import math
from config import Config

# 创建全局配置对象
config = Config()

class FasterRCNN(nn.Module):
    """
    Faster R-CNN 模型的主体架构
    
    参数:
        num_classes (int): 类别数量（包括背景）
        min_size (int): 输入图像的最小边长
        max_size (int): 输入图像的最大边长
    """
    def __init__(self, num_classes, min_size=800, max_size=1333):
        super(FasterRCNN, self).__init__()
        self.min_size = min_size
        self.max_size = max_size
        
        # 使用预训练的ResNet50作为backbone，移除最后两层
        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # 特征金字塔网络，用于生成多尺度特征图
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],  # ResNet各阶段输出通道数
            out_channels=256  # FPN统一输出通道数
        )
        
        # 区域提议网络，用于生成候选框
        self.rpn = RegionProposalNetwork(
            anchor_sizes=((32,), (64,), (128,), (256,), (512,)),  # 每层特征图的anchor大小
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,  # anchor的宽高比
            nms_thresh=config.RPN_NMS_THRESH,
            pre_nms_top_n=config.RPN_PRE_NMS_TOP_N,
            post_nms_top_n=config.RPN_POST_NMS_TOP_N
        )
        
        # ROI处理头部，用于分类和边界框回归
        self.roi_head = RoIHead(
            num_classes=num_classes,
            box_roi_pool=MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # 使用的特征图层级
                output_size=7,  # ROI Align输出大小
                sampling_ratio=2  # 采样点数量
            ),
            fg_iou_thresh=config.ROI_FG_IOU_THRESH,
            bg_iou_thresh=config.ROI_BG_IOU_THRESH,
            batch_size_per_image=config.ROI_BATCH_SIZE_PER_IMAGE,
            positive_fraction=config.ROI_POSITIVE_FRACTION
        )
        
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
            
        original_image_sizes = []
        for img in images:
            original_image_sizes.append((img.shape[-2], img.shape[-1]))
            
        # 提取特征
        features = self.backbone(images)
        features = self.fpn(features)
        
        # RPN前向传播
        proposals, proposal_losses = self.rpn(images, features, targets)
        
        # ROI head前向传播
        detections, detector_losses = self.roi_head(features, proposals, targets)
        
        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)
        
        if self.training:
            return losses
            
        return detections
