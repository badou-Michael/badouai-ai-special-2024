import cv2
import numpy as np

# 读取图像
img = cv2.imread('lenna.png')  # 导入lenna图

# 实现均值哈希（aHash）
gray_aHash = cv2.cvtColor(cv2.resize(img, (8, 8)), cv2.COLOR_BGR2GRAY)
avg = np.mean(gray_aHash)
aHash = ''.join(['1' if pixel > avg else '0' for pixel in gray_aHash.flatten()])
print("aHash:", aHash)

# 实现差值哈希（dHash）
gray_dHash = cv2.cvtColor(cv2.resize(img, (9, 8)), cv2.COLOR_BGR2GRAY)
dHash = ''.join(['1' if gray_dHash[i, j] > gray_dHash[i, j + 1] else '0' for i in range(8) for j in range(8)])
print("dHash:", dHash)
