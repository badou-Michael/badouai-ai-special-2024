import cv2
import numpy as np
import math

# 实现最临近插值
img = cv2.imread('lenna.png')
src_h, src_w, channel = img.shape
dst_h, dst_w = 1024, 1024
dst = np.zeros([dst_h, dst_w, channel], dtype=np.uint8)
for x_dst in range(dst_w):
    x_src = (x_dst + 0.5) * (src_w / dst_w) if (x_dst + 0.5) * (src_w / dst_w) <= src_w-1 else src_w-1
    for y_dst in range(dst_h):
        y_src = (y_dst + 0.5) * (src_h / dst_h) if (y_dst + 0.5) * (src_h / dst_h) <= src_h-1 else src_h-1
        # 四舍五入，将原图上对应的像素点的像素值赋给目标图的像素点
        dst[y_dst, x_dst] = img[round(y_src), round(x_src)]
cv2.imshow("Lenna", img)
cv2.imshow("Lenna after nearest interpolation", dst)
cv2.waitKey(0)


# 实现双线性插值
img = cv2.imread('lenna.png')
src_h, src_w, channel = img.shape
dst_h, dst_w = 1024, 1024
dst = np.zeros([dst_h, dst_w, channel], dtype=np.uint8)
for x_dst in range(dst_w):
    x_src = (x_dst + 0.5) * (src_w / dst_w) if (x_dst + 0.5) * (src_w / dst_w) <= src_w-1 else src_w-1
    for y_dst in range(dst_h):
        y_src = (y_dst + 0.5) * (src_h / dst_h) if (y_dst + 0.5) * (src_h / dst_h) <= src_h-1 else src_h-1
        # 寻找目标点所对应的原图上的点四周的四个点
        x1, x2, y1, y2 = [math.floor(x_src), math.ceil(x_src), math.floor(y_src), math.ceil(y_src)]
        dst[y_dst, x_dst] = (y2 - y_src)*((x2 - x_src)*img[y1, x1] + (x_src - x1) * img[y1, x2]) + (y_src - y1)*((x2 - x_src) * img[y2, x1] + (x_src - x1) * img[y2, x2])
cv2.imshow("Lenna", img)
cv2.imshow("Lenna after nearest interpolation", dst)
cv2.waitKey(0)

