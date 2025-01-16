import numpy as np
import cv2
import random

def random_colors(N):
    """生成N个随机颜色"""
    np.random.seed(1)
    colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(N)]
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """将掩码应用到图像上"""
    # 确保图像类型为uint8
    image = image.astype(np.uint8)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                 image[:, :, c] * (1 - alpha) + alpha * color[c],
                                 image[:, :, c])
    return image

def display_instances(image, boxes, masks, scores, class_names, score_threshold=0.7):
    """显示检测结果"""
    # 确保图像类型为uint8
    image = image.astype(np.uint8).copy()
    
    # 生成随机颜色
    colors = random_colors(len(class_names))
    
    # 遍历每个实例
    for i in range(boxes.shape[0]):
        if scores[i] < score_threshold:  # 分数阈值
            continue
            
        # 边界框
        y1, x1, y2, x2 = boxes[i].astype(np.int32)
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[i], 2)
        
        # 掩码
        mask = masks[i].astype(np.bool_)
        image = apply_mask(image, mask, colors[i])
        
        # 标签
        label = class_names[i]
        score = scores[i]
        caption = f'{label} {score:.2f}'
        cv2.putText(image, caption, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
    
    # 显示结果
    cv2.imshow('Result', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    cv2.imwrite('result.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) 