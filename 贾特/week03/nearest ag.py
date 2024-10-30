import cv2
import numpy as np

def function(img):
    height, width, channels = img.shape
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    sh = height / 800
    sw = width / 800
    for i in range(800):
        for j in range(800):
            x = int(i * sh + 0.5)  # int(),转为整型，使用向下取整。
            y = int(j * sw + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage


img = cv2.imread("lenna.png")
net = function(img)
print(net)
print(net.shape)
cv2.imshow("nearest interp", net)
cv2.imshow("src image", img)
cv2.waitKey(0)
