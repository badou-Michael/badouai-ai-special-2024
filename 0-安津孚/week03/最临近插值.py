import cv2
import numpy as np


# 最近邻插值方法实现图像缩放

def function(img, size):
    height, width, channels = img.shape
    print(img.shape)
    emptyImage = np.zeros((size, size, channels), np.uint8)
    sh = size / height
    sw = size / width
    for i in range(size):
        for j in range(size):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage


# 读取图像
img = cv2.imread("../week02/lenna.png")
zoom = function(img, 800)

# 使用 OpenCV 的 resize 函数进行缩放
# zoom = cv2.resize(img, (800, 800), interpolation=cv2.INTER_NEAREST)
print("zoom", zoom)
print("zoom", zoom.shape)

# 显示图像
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)

# 等待按键
cv2.waitKey(0)
cv2.destroyAllWindows()
