import numpy as np
import cv2

# cv2读取图片
img = cv2.imread("../lenna.png")

# 提取读入图片信息
h, w, c = img.shape

# 创建空图像 目标放大为800*800
emptyImg = np.zeros((800, 800, c), img.dtype)
# emptyImg = np.zeros((800, 800, c),np.uint8)

# 计算目标图片放大倍数
# 这里需要注意放大的计算，否者会出现超出（512,512）的范围，导致报错
sh = 800/h
sw = 800/h

for i in range(800):
    for j in range(800):
        # int 向下取整
        oldH = int(i/sh + 0.5)
        oldW = int(j/sw + 0.5)
        emptyImg[i, j] = img[oldH, oldW]

cv2.imshow("nearstImg", emptyImg)
cv2.imshow("CV2Gray", img)
# 停留显示
cv2.waitKey(0)
