import torch
import torch.nn as nn
import torch.nn.functional as F

class RPNLoss(nn.Module):
    def __init__(self, config):
        super(RPNLoss, self).__init__()
        self.config = config
        
    def forward(self, rpn_class_logits, rpn_bbox_pred, rpn_labels, rpn_bbox_targets):
        """RPN损失计算"""
        if rpn_labels.numel() == 0:
            return torch.tensor(0.0).to(rpn_labels.device), torch.tensor(0.0).to(rpn_labels.device)
        
        # 添加正负样本平衡
        pos_mask = rpn_labels == 1
        neg_mask = rpn_labels == 0
        num_pos = pos_mask.sum().item()
        num_neg = neg_mask.sum().item()
        
        # 平衡系数
        pos_weight = 1.0
        neg_weight = num_pos / max(1, num_neg)
        
        # 分类损失
        rpn_class_loss = F.cross_entropy(
            rpn_class_logits, rpn_labels,
            ignore_index=-1
        )
        
        # 回归损失
        rpn_bbox_loss = F.smooth_l1_loss(
            rpn_bbox_pred,
            rpn_bbox_targets,
            reduction='mean',
            beta=1.0/9.0
        )
        
        return rpn_class_loss, rpn_bbox_loss

class DetectionLoss(nn.Module):
    def __init__(self, config):
        super(DetectionLoss, self).__init__()
        self.config = config
        
    def forward(self, class_logits, box_regression, labels, regression_targets):
        """检测头损失计算"""
        if labels.numel() == 0:
            return (torch.tensor(0.0).to(labels.device), 
                   torch.tensor(0.0).to(labels.device))
        
        # 分类损失
        classification_loss = F.cross_entropy(
            class_logits, labels
        )
        
        # 只对正样本计算回归损失
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        
        box_regression_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            reduction='mean',
            beta=1.0
        )
        
        return classification_loss, box_regression_loss

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
        
    def forward(self, mask_logits, mask_targets):
        """Mask分支损失计算"""
        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits, mask_targets
        )
        return mask_loss 