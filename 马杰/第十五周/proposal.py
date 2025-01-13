import torch
import torchvision.ops as ops
from .box_utils import clip_boxes_to_image, box_transform_inv

def generate_proposals(rpn_bbox_pred, rpn_cls_prob, anchors, image_shape, config):
    """生成RPN proposals"""
    batch_size = rpn_bbox_pred.shape[0]
    proposals_list = []
    scores_list = []
    
    for i in range(batch_size):
        props, scores = _generate_proposals_single_image(
            rpn_bbox_pred[i], rpn_cls_prob[i], anchors, image_shape, config)
        proposals_list.append(props)
        scores_list.append(scores)
    
    return proposals_list, scores_list

def _generate_proposals_single_image(rpn_bbox_pred, rpn_cls_prob, anchors, image_shape, config):
    """生成单张图像的RPN proposals"""
    # 获取预测scores
    scores = rpn_cls_prob[:, 1]
    
    # 将预测的bbox deltas应用到anchors上
    proposals = box_transform_inv(anchors, rpn_bbox_pred)
    
    # 裁剪到图像范围内
    proposals = clip_boxes_to_image(proposals, image_shape)
    
    # 移除小框
    keep = ops.remove_small_boxes(proposals, config.RPN_MIN_SIZE)
    proposals = proposals[keep]
    scores = scores[keep]
    
    # NMS
    keep = ops.nms(
        proposals,
        scores,
        config.RPN_NMS_THRESHOLD
    )
    
    # 取前N个
    keep = keep[:config.POST_NMS_ROIS_TRAINING]
    proposals = proposals[keep]
    scores = scores[keep]
    
    if proposals.numel() == 0:
        # 返回空proposals
        return (
            torch.zeros((0, 4), device=proposals.device),
            torch.zeros(0, device=proposals.device)
        )
    
    return proposals, scores 