'''
最邻近插值算法()
采样：用多大的像素描述图像
量化：用多大范围的数值来表示采样的像素点
假设我们放大一张512x512图像,计算出缩放比例，通过缩放比例计算目标像素在源像素对应位置
如果计算的结果为整数那就一一对应 ，为小数时采用四舍五入的方式取整，最后赋值目标像素。
优点： 计算简单
缺点：目标像素直接取邻近的像素值 ，没有考虑这点是否可取，也就是是否连续而不是离散的
'''
import cv2 as cv

import numpy as np
import math
def zoom_func(img):
    height,width,channel = img.shape    #源图像三个属性，h,w,c
    new_emptyImage  = np.zeros((800,800,channel),np.uint8)     #800x800 空图像,每个像素值都是0
    hk=height/800   #高度缩放比例
    wk=width/800    #宽度缩放比例
    for i in range(800):
        for j in range(800):
            # x=math.floor(i/hk)    #math模块的四舍五入
            # y=math.floor(j/wk)
            #print(x,y)
            x=int(i/hk+0.5)   #int 默认会作向下取整
            y=int(j/wk+0.5)
            new_emptyImage[i,j] = img[x,y]
    return new_emptyImage



#读图片
img = cv.imread("./images/lenna.png")    # 512x512
#zoom = zoom_func(img)
#cv2提供的最邻近插值接口
zoom = cv.resize(img,(800,800),interpolation=cv.INTER_NEAREST)
print(img)
print(img.shape)
print("----------------")
print(zoom)
print(zoom.shape)
cv.imshow("src_image",img)
cv.imshow("new_image",zoom)
cv.waitKey(0)
