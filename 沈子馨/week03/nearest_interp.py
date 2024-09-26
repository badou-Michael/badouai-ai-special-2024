import cv2
import numpy as np

def function(img):
    height, width, channel = img.shape
    # np.zeros()初始化函数
    emptyImage = np.zeros((800, 800, channel), dtype=np.uint8)  #np.uint8 无符号整型0-255
    sh = 800/height   #比例
    sw = 800/width
    for i in range(800):
        for j in range(800):
            x = int(i/sh+0.5)  #+0.5为了向下取整的精度
            y = int(j/sw+0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage

img = cv2.imread("lenna.png")
nearestImage = function(img)
print(nearestImage)
print(nearestImage.shape)
cv2.imshow("nearest interp", nearestImage)
cv2.imshow("image", img)
cv2.waitKey(0)
