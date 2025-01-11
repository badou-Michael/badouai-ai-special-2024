import cv2
import numpy as np


def function(img):
    height, width, channels = img.shape
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    sh = 800 / height
    sw = 800 / width
    for i in range(800):
        for j in range(800):
            x = int(i / sh + 0.5)  # int(),转为整型，使用向下取整。，所以 + 0.5 可以模拟四舍五入
            y = int(j / sw + 0.5)  # 找的就是放大后的像素点距离原来图像中四个较近的像素点中最近的一个的像素值
            emptyImage[i, j] = img[x, y] #找到原图该点后直接赋值过去就行
    return emptyImage
#带来缺点：只考虑最近的 而不考虑另外三个

# cv2.resize(img, (800,800,c),near/bin)

img = cv2.imread("lenna.png")
zoom = function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)


