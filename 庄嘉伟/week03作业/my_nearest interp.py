#最临近插值算法
#任何需要计算虚拟点像素值的情况都可以使用该算法

import cv2
import numpy as np

def function(img):#手写方法
    height,width,channels = img.shape
    enptyImage = np.zeros((800,800,channels),np.uint8) #创建一个新的空白图片，规格为800*800，uint8为无符号八位整数
    sh = 800/height
    sw = 800/width
    for i in range(800):
        for j in range(800):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            enptyImage[i,j] = img[x,y]

    return enptyImage
#调用接口
# cv2.resize(img, (800,800,c),near/bin) #near/bin为选择最临近插值或双线性插值，伪代码

def cv_show(name,img):   #展示函数
    cv2.imshow(name,img)
    cv2.waitKey(0)  #窗口等待
    cv2.destroyWindow()  #窗口销毁
img = cv2.imread("lenna.png")  #加载图片
zoom = function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("image", img)
cv2.imshow("nearest interp", zoom)
cv2.waitKey(0)
cv2.destroyWindow()
# cv_show("image",img)
# cv_show("nearest interp",zoom)



