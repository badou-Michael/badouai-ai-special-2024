import numpy as np
import cv2


def near_interpolation(img, target_height, target_width):
    # 获取源图像的高和宽
    original_height, original_width = img.shape[:2]
    # 创建一个新图像，大小为目标图像大小
    dst_img = np.zeros((target_height, target_width, 3), np.uint8)
    # 计算缩放比例
    scale_h = original_height / target_height
    scale_w = original_width / target_width
    for dst_y in range(target_height):
        for dst_x in range(target_width):
            # 计算源图像坐标
            src_y = int(dst_y * scale_h)
            src_x = int(dst_x * scale_w)
            src_y = min(src_y, original_height - 1)
            src_x = min(src_x, original_width - 1)
            # 复制像素值
            dst_img[dst_y, dst_x] = img[src_y, src_x]
    return dst_img

img = cv2.imread('lenna.png')
if img is None:
    print('Failed to load image')
    exit()
dst_img = near_interpolation(img, 128, 128)
cv2.imshow('original', img)
cv2.imshow('near_interpolation', dst_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
