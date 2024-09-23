"""
插值，yzc
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
####1.双线性插值  --resize
imgLinear = cv2.resize(img,(700,700),interpolation=cv2.INTER_LINEAR)
cv2.imshow("linear resize Img：",imgLinear)
# cv2.waitKey()

####------------------------------------------------####
####2.双线性插值  --推导过程

def linearInter(img,dim):
    srcH,srcW,c = img.shape
    dstH,dstW=dim[0],dim[1]
    if srcH == dstH and srcW == dstW :
        return img.copy()
    kongImg = np.zeros((dstH,dstW,c),dtype=np.uint8)
    calc_x , calc_y = float(dstW)/srcW , float(dstH)/srcH
    for i in range(c):
        for dst_y in range(dstH):
            for dst_x in range(dstW):
                #中心对齐
                src_x = (dst_x +0.5)/calc_x - 0.5
                src_y = (dst_y +0.5)/calc_y - 0.5
                #边界值处理
                src_x0 = int(src_x)
                src_x1 = min(src_x0 + 1 , srcW -1)
                src_y0 = int(src_y)
                src_y1 = min(src_y0 + 1 ,srcW -1)
                ##计算
                R1 = (src_x1 - src_x)* img[src_x0,src_y0,i] + (src_x - src_x0)* img[src_x1,src_y0,i]
                R2 = (src_x1 - src_x)* img[src_x0,src_y1,i] + (src_x - src_x0)* img[src_x1,src_y1,i]

                kongImg[dst_x,dst_y,i]= (src_y1 - src_y) * R1 +(src_y - src_y0) * R2
    return kongImg


img_result = linearInter(img,(700,700))
cv2.imshow("yzc img linear:",img_result)
cv2.waitKey()
