# 最邻近插值法
import cv2
import numpy as np


def function(img):
    height, width, channels = img.shape
    empty_img = np.zeros((800, 800, channels), np.uint8)
    sh = 800/height
    sw = 800/width
    for i in range(800):
        for j in range(800):
            x = int(i/sh+0.5)  # 缩放比例
            y = int(j/sw+0.5)
            if x != i and y !=j:
                print(f"x={x}, i={i}, y={y}, j={j}")
            empty_img[i, j] = img[x, y]

    return empty_img


img = cv2.imread("lenna.png")
zoom = function(img)
print(zoom)
print(zoom.shape)

cv2.imshow("nearest_interp_image", zoom)
cv2.imshow("origin_image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

