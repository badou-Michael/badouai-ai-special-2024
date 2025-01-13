import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.ops as ops
from utils.anchors import generate_anchors
from utils.losses import RPNLoss, DetectionLoss, MaskLoss

class ResNet50FPN(nn.Module):
    def __init__(self):
        super(ResNet50FPN, self).__init__()
        # 使用预训练的ResNet50作为backbone
        self.resnet = torchvision.models.resnet50(pretrained=True)
        # FPN部分
        self.fpn_c2p2 = nn.Conv2d(256, 256, 1)
        self.fpn_c3p3 = nn.Conv2d(512, 256, 1)
        self.fpn_c4p4 = nn.Conv2d(1024, 256, 1)
        self.fpn_c5p5 = nn.Conv2d(2048, 256, 1)
        
        # 横向连接
        self.fpn_p2 = nn.Conv2d(256, 256, 3, padding=1)
        self.fpn_p3 = nn.Conv2d(256, 256, 3, padding=1)
        self.fpn_p4 = nn.Conv2d(256, 256, 3, padding=1)
        self.fpn_p5 = nn.Conv2d(256, 256, 3, padding=1)
        self.fpn_p6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)

    def forward(self, x):
        # ResNet stages
        c1 = self.resnet.conv1(x)
        c1 = self.resnet.bn1(c1)
        c1 = self.resnet.relu(c1)
        c1 = self.resnet.maxpool(c1)
        
        c2 = self.resnet.layer1(c1)
        c3 = self.resnet.layer2(c2)
        c4 = self.resnet.layer3(c3)
        c5 = self.resnet.layer4(c4)
        
        # FPN自顶向下路径
        p5 = self.fpn_c5p5(c5)
        p4 = self.fpn_c4p4(c4) + F.interpolate(p5, scale_factor=2)
        p3 = self.fpn_c3p3(c3) + F.interpolate(p4, scale_factor=2)
        p2 = self.fpn_c2p2(c2) + F.interpolate(p3, scale_factor=2)
        
        # 最终的特征图
        p2 = self.fpn_p2(p2)
        p3 = self.fpn_p3(p3)
        p4 = self.fpn_p4(p4)
        p5 = self.fpn_p5(p5)
        p6 = self.fpn_p6(p5)
        
        return [p2, p3, p4, p5, p6]

class RPN(nn.Module):
    def __init__(self, anchor_scales, anchor_ratios, feature_channels=256):
        super(RPN, self).__init__()
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.num_anchors = len(anchor_scales) * len(anchor_ratios)
        self.feature_stride = 16
        self.anchor_stride = 1
        
        # RPN卷积层
        self.conv = nn.Conv2d(feature_channels, feature_channels, 3, padding=1)
        # 分类分支
        self.cls_logits = nn.Conv2d(feature_channels, self.num_anchors * 2, 1)
        # 回归分支
        self.bbox_pred = nn.Conv2d(feature_channels, self.num_anchors * 4, 1)
        
    def forward(self, x):
        # x是FPN的特征图列表
        rpn_logits = []
        rpn_bbox = []
        
        for feature in x:
            t = F.relu(self.conv(feature))
            rpn_logits.append(self.cls_logits(t))
            rpn_bbox.append(self.bbox_pred(t))
            
        return rpn_logits, rpn_bbox

    def generate_anchors(self, features):
        """为每个特征层生成anchors"""
        anchors = []
        for feature_map in features:
            feature_shape = feature_map.shape[-2:]
            anchors_for_level = generate_anchors(
                self.anchor_scales,
                self.anchor_ratios,
                feature_shape,
                self.feature_stride,
                self.anchor_stride
            )
            anchors.append(anchors_for_level)
        return anchors

class MaskRCNN(nn.Module):
    def __init__(self, num_classes, config):
        super(MaskRCNN, self).__init__()
        self.num_classes = num_classes
        self.config = config
        self.feature_stride = config.FEATURE_STRIDE
        
        # Backbone + FPN
        self.backbone = ResNet50FPN()
        
        # RPN
        self.rpn = RPN(config.RPN_ANCHOR_SCALES, 
                      config.RPN_ANCHOR_RATIOS)
        
        # Box head
        self.box_head = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        
        # Box predictor
        self.box_predictor = nn.Linear(1024, num_classes * 4)
        self.cls_predictor = nn.Linear(1024, num_classes)
        
        # Mask head
        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 2, 2),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
        
        # 添加损失函数
        self.rpn_loss = RPNLoss(config)
        self.detection_loss = DetectionLoss(config)
        self.mask_loss = MaskLoss()
        
    def forward(self, images, targets=None):
        # 特征提取
        features = self.backbone(images)
        
        # RPN预测
        rpn_logits, rpn_bbox = self.rpn(features)
        
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            # 确保targets是字典列表
            targets = [{k: v for k, v in t.items()} for t in targets]
            # 训练模式
            anchors = self.rpn.generate_anchors(features)
            proposals, proposal_scores = self.generate_proposals(rpn_bbox, rpn_logits, anchors, images.shape)
            # ROI Pooling
            roi_features = self.roi_align(features, proposals)
            
            # Box head
            box_features = self.box_head(roi_features.view(roi_features.shape[0], -1))
            
            # 预测
            box_deltas = self.box_predictor(box_features)
            class_logits = self.cls_predictor(box_features)
            
            # Mask head
            mask_features = self.roi_align(features, proposals, 14)
            mask_logits = self.mask_head(mask_features)
            
            # 计算RPN损失
            rpn_class_loss, rpn_box_loss = self.rpn_loss(
                rpn_logits, rpn_bbox, 
                [t['rpn_labels'] for t in targets], 
                [t['rpn_bbox_targets'] for t in targets]
            )
            
            # 计算检测头损失
            class_loss, box_loss = self.detection_loss(
                class_logits, box_deltas, 
                [t['labels'] for t in targets], 
                [t['box_targets'] for t in targets]
            )
            
            # 计算mask损失
            mask_loss = self.mask_loss(
                mask_logits, 
                [t['mask_targets'] for t in targets]
            )
            
            losses = {
                'loss_rpn_cls': rpn_class_loss,
                'loss_rpn_box': rpn_box_loss,
                'loss_cls': class_loss,
                'loss_box': box_loss,
                'loss_mask': mask_loss
            }
            return losses
        else:
            # 推理模式
            anchors = self.rpn.generate_anchors(features)
            proposals = self.generate_proposals(rpn_bbox, rpn_logits, anchors, images.shape)
            roi_features = self.roi_align(features, proposals)
            
            box_features = self.box_head(roi_features.view(roi_features.shape[0], -1))
            box_deltas = self.box_predictor(box_features)
            class_logits = self.cls_predictor(box_features)
            
            # 后处理
            boxes = self.apply_deltas(proposals, box_deltas)
            scores = F.softmax(class_logits, dim=-1)
            
            # NMS
            keep = torchvision.ops.nms(boxes, scores.max(1)[0], self.config.NMS_THRESHOLD)
            
            # Mask prediction
            mask_features = self.roi_align(features, boxes[keep], 14)
            mask_logits = self.mask_head(mask_features)
            
            return {
                'boxes': boxes[keep],
                'scores': scores[keep],
                'masks': mask_logits[keep]
            } 

    def generate_proposals(self, rpn_bbox_pred, rpn_cls_prob, anchors, image_shape):
        """生成proposals"""
        proposals = []
        scores = []
        for box_delta, cls_prob, anchor in zip(rpn_bbox_pred, rpn_cls_prob, anchors):
            prop, score = generate_proposals(box_delta, cls_prob, anchor, image_shape, self.config)
            proposals.append(prop)
            scores.append(score)
        return proposals, scores

    def roi_align(self, features, boxes, output_size):
        """ROI Align操作"""
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        # 需要处理feature pyramid
        rois = []
        for level, feature in enumerate(features):
            roi = ops.roi_align(
                feature,
                boxes,
                output_size,
                spatial_scale=1.0/(self.feature_stride * (2**level)),
                sampling_ratio=2
            )
            rois.append(roi)
        return torch.cat(rois, dim=0)

    def initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0) 

    def apply_deltas(self, boxes, deltas):
        """添加缺失的方法"""
        # boxes: [N, 4], deltas: [N, num_classes * 4]
        boxes = boxes.unsqueeze(1).repeat(1, self.num_classes, 1)
        boxes = boxes.view(-1, 4)
        deltas = deltas.view(-1, 4)
        
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        
        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]
        
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        
        pred_boxes = torch.zeros_like(boxes)
        pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
        
        return pred_boxes.view(-1, self.num_classes, 4) 