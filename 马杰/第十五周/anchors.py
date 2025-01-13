import numpy as np

def generate_anchors(scales, ratios, feature_shape, feature_stride, anchor_stride):
    """生成anchors"""
    # 获取所有可能的尺度和比例组合
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()
    
    # 计算高度和宽度
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)
    
    # 生成中心点
    shifts_y = np.arange(0, feature_shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, feature_shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
    
    # 组合所有anchors
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])
    
    # 转换为[y1, x1, y2, x2]格式
    boxes = np.concatenate([
        box_centers - 0.5 * box_sizes,
        box_centers + 0.5 * box_sizes
    ], axis=1)
    
    return boxes 