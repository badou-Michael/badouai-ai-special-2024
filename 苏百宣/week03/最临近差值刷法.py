#实现最临近差值算法
#author：苏百宣

import cv2
import numpy as np

def function(img):
    height,width,channels = img.shape
    emptyImage = np.zeros((800,800,channels),np.uint8)
    sh = 800/height
    sw = 800/width
    for i in range(800):
        for j in range (800):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            emptyImage[i,j] = img[x,y]
    return emptyImage

img = cv2.imread("lenna.png")
# 检查图像是否加载成功
if img is None:
    print("图像加载失败，请检查文件路径")
    exit()

zoom = function(img)
print(zoom)
print(zoom.shape)

# 显示图像并设置窗口位置
cv2.imshow("原图", img)
cv2.moveWindow("原图", 100, 100)  # 设置原图窗口位置
cv2.imshow("放大图", zoom)
cv2.moveWindow("放大图", 900, 100)  # 设置放大图窗口位置

cv2.waitKey(0)  # 等待按键关闭窗口
cv2.destroyAllWindows()  # 关闭所有窗口
