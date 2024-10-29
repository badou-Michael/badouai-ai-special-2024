#实现最临近插值

import cv2

img = cv2.imread("lenna.png")

# 使用最近邻插值放大图像
zoom = cv2.resize(img, (800, 800), interpolation=cv2.INTER_NEAREST)

# 打印
print(zoom)
print(zoom.shape)

# 显示放大后的图像和原图
cv2.imshow("Nearest Image", zoom)
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#实现双线性插值

import cv2

img = cv2.imread('lenna.png')

# 使用双线性插值放大图像
dst = cv2.resize(img, (700, 700), interpolation=cv2.INTER_LINEAR)

# 显示图像
cv2.imshow('bilinear interp', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
