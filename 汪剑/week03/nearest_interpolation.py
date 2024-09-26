import cv2
import numpy as np
import matplotlib.pyplot as plt

#方法一
# def fn_nearest_interpolation(filename,h,w):
#     img = cv2.imread(filename)
#     height,wide,channel = img.shape
#     TransformImage = np.zeros((h,w,channel),dtype=np.uint8)
#     sh = h/height
#     sw = w/wide
#     for i in range(h):
#         for j in range(w):
#             x = round(i/sh)
#             y = round(j/sw)
#             TransformImage[i,j] = img[x,y]
#     return TransformImage
#
#
# TransformImage = fn_nearest_interpolation('lenna.png',800,800)
# cv2.imshow('TransformImage：',TransformImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 方法二
img = cv2.imread('lenna.png')

"""
cv2.INTER_NEAREST：最近邻插值（速度最快，但可能出现马赛克效果）
cv2.INTER_LINEAR：双线性插值（默认值，效果较好，速度适中）
cv2.INTER_CUBIC：三次插值（效果更好，适合缩小图像，但速度较慢）
cv2.INTER_LANCZOS4：Lanczos插值（效果非常好，适合缩小图像，速度较慢）
cv2.INTER_AREA：区域插值（适合缩小图像）
"""
TransformImage = cv2.resize(img, (800, 800), interpolation=cv2.INTER_NEAREST)
# TransformImage1 = cv2.resize(img, None, fx=800 / 512, fy=800 / 512, interpolation=cv2.INTER_NEAREST)

cv2.imshow('TransformImage：', TransformImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# img = cv2.resize(img, dsize=(800, 800))
# Image = np.hstack((img,TransformImage))
# cv2.imshow('Image：',Image)
# cv2.waitKey(0)
