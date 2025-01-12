import torch
import torchvision.ops as ops

def roi_align(features, boxes, output_size):
    """ROI Align操作"""
    return ops.roi_align(features, boxes, output_size,
                        spatial_scale=1.0/16.0, sampling_ratio=2) 