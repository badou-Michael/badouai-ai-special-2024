import cv2
import numpy as np

# def linear_interpolation(img, scale_percent):
#     h, w, c = img.shape
#     if scale_percent == 1:
#         return img
#     target_image = np.zeros((int(h * scale_percent), int(w * scale_percent), c), np.uint8)
#     for i in range(c):
#         for dst_y in range(int(h * scale_percent)):
#             for dst_x in range(int(w * scale_percent)):
#                 #几何中心对齐
#                 src_x = (dst_x + 0.5) / scale_percent - 0.5
#                 src_y = (dst_y + 0.5) / scale_percent - 0.5
#                 #找到邻近四个点
#                 src_x0 = int(src_x)
#                 src_x1 = min(src_x0 + 1, w - 1)
#                 src_y0 = int(src_y)
#                 src_y1 = min(src_y0 + 1, h - 1)
#                 #两个x方向插值，一个y方向插值
#                 inter1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
#                 inter2 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
#                 target_image[dst_y, dst_x, i] = int((src_y1 - src_y) * inter2 + (src_y - src_y0) * inter1)
#
#     return target_image

img = cv2.imread('lenna.png')
scale_percent = int(input('请输入缩放比例（整数）：')) / 100
target_image = cv2.resize(img, (int(img.shape[0] * scale_percent), int(img.shape[1] * scale_percent)), interpolation = cv2.INTER_LINEAR)
# target_image = linear_interpolation(img, scale_percent)
cv2.imshow('lenna image', img)
print(img)
print(img.shape)
cv2.imshow('target image', target_image)
print(target_image)
print(target_image.shape)
cv2.waitKey(0)
