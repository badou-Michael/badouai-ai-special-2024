import numpy as np
import cv2

# 设置均值和标准差
mean = 0
std_dev = 1

# 生成一个高斯随机数
random_number = np.random.normal(mean, std_dev)
print(random_number)
ctl = lambda num: max(0, min(255, num))
img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
print(type(img))
# h, w  = img.shape[:2]
# img_guass = np.zeros([h, w], img.dtype)
# img1 = np.zeros([h, w], img.dtype)
# for i in range(h):
#     for j in range(w):
#         img_guass[i][j] = ctl(img[i][j] + random_number)
img_guass = np.floor(np.clip(img + random_number, 0, 255)).astype(img.dtype)
print(type(img_guass))

cv2.imshow('original', img)
cv2.imshow('guass', img_guass)
cv2.waitKey(0)



