import cv2
import numpy as np


def doubleLinesInsert(img, target_height, target_width):
    original_height, original_width, channels = img.shape
    print('dtype:', img.dtype)
    if original_height == target_height and original_width == target_width:
        print('返回原图')
        return img
    dst_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    # 计算缩放比例
    scale_height = original_height / target_height
    scale_width = original_width / target_width
    for dst_y in range(target_height):
        for dst_x in range(target_width):
            #计算源图像的位置 +0.5实现中心对齐
            src_y = (dst_y + 0.5) * scale_height - 0.5
            src_x = (dst_x + 0.5) * scale_width - 0.5

            # 计算四个邻域像素的位置
            src_x0 = int(src_x)
            src_y0 = int(src_y)
            src_x1 = min(src_x0 + 1, original_width - 1)
            src_y1 = min(src_y0 + 1, original_height - 1)

            # 计算权重
            for c in range(channels):
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, c] + (src_x - src_x0) * img[src_y0, src_x1, c]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, c] + (src_x - src_x0) * img[src_y1, src_x1, c]
                # dst_img[dst_y, dst_x, c] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
                dst_img[dst_y, dst_x, c] = np.uint8((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
                # dst_img[dst_y, dst_x, c] = (src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1
    return dst_img


img = cv2.imread('lenna.png')
if img is None:
    print('Failed to load image')
    exit()
new_img = shuangxian(img, 700, 700)
cv2.imshow('original', img)
cv2.imshow('new', new_img)
cv2.waitKey(0)
