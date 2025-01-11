from idlelib.configdialog import changes

import cv2
import numpy as np
'''
邻近插值图片处理优化
'''
def function(img):
    # 获取图像的形状信息
    height,width,channel = img.shape

    # 创建一张目标图像存在倍率关系的图像用于放大缩小，1000*1000尺寸
    image_empty= np.zeros((1000,1000,channel),np.uint8)
    # 目标图像和元数据的高宽比
    sh = 1000/height
    sw = 1000/width
    # 目标图像数据循环v
    for i in range(1000):
        for j in  range(1000):
            x = int(i/sh+0.5)
            y = int(j/sw+0.5)
            # 对目标图像的数据进行映射
            image_empty[i,j] = img[x,y]
    return image_empty
img= cv2.imread("../../request/task2/lenna.png")
zoom = function(img) # 调用函数并存储结果到变量zoom中
print(zoom)# 输出zoom值
print(zoom.shape)# 输出zoom的shape矩阵数据
cv2.imshow("nearest_img",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)


