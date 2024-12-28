import cv2
import numpy as np
#找到原图大小
#找到缩放比例
#初始化目标图
#找到目标图每个像素对应原图最邻近像素，并赋像素值
# def nearest_interpolation(img, scale_percent):
#     h, w, c = img.shape
#     target_image = np.zeros((int(h * scale_percent), int(w * scale_percent), c), np.uint8)
#     for i in range(int(h * scale_percent)):
#         for j in range(int(w * scale_percent)):
#             x = int(i / scale_percent + 0.5)
#             y = int(j / scale_percent + 0.5)
#             target_image[i, j] = img[x, y]
#     return target_image

img = cv2.imread('lenna.png')
scale = int(input('请输入缩放比例（整十）：'))
scale_percent = scale / 100
target_image = cv2.resize(img, (int(img.shape[0] * scale_percent), int(img.shape[1] * scale_percent)), interpolation = cv2.INTER_NEAREST)
# target_image = nearest_interpolation(img, scale_percent)
cv2.imshow('lenna image', img)
print(img)
print(img.shape)
cv2.imshow('target image',target_image)
print(target_image)
print(target_image.shape)
cv2.waitKey(0)
