import cv2
import numpy as np
# 手写最临近插值
def function(img):
    height,width,channels = img.shape
    emptyImage = np.zeros((800,800,channels), np.uint8)
    sh = 800/height      # 缩放比例
    sw = 800/width       # 缩放比例
    for i in range(800):
        for j in range(800):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage

# cv2.resize(img, (800,800, c), near/bin)    直接调用第三方接口计算最临近插值

img = cv2.imread("lenna.png")
zoom = function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image", img)
cv2.waitKey(0)      # 展示图像，不然图像一闪而过
