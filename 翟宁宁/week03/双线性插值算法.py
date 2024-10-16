'''
双线性插值
目标图像在原图像对应的虚拟位置（x,y)，该点的像素值与邻近四个点都有贡献 ，贡献多少为权重

'''

import cv2  as cv
import numpy as np
import  math



def billnear_func(img, out_img):
    src_w,src_h,c=img.shape
    out_w,out_h=out_img[0],out_img[1]
    target_img = np.zeros((out_w,out_h,c),dtype=np.uint8)   #目标图像
    w_k = src_w/out_w  #横向缩放比 w_k
    h_k = src_h/out_h  #纵向缩放比 h_k
    for i in range(c):
        for t_x in range(out_w):
            for t_y in range(out_h):
                # 目标图到原图映射
                src_x = (t_x+0.5)  * (w_k-0.5)
                src_y = (t_y+0.5) * (h_k-0.5)

                # 边界值
                src_x0 = int(math.floor(src_x))
                src_x1 = min(src_x0+1,src_w-1)
                src_y0 = int(math.floor(src_y))
                src_y1 = min(src_y0+1,src_h-1)

                # 带入公式插值
                r1 = (src_x1-src_x)*img[src_x0,src_y0,i] + (src_x-src_x0)*img[src_x1,src_y0,i]
                r2 = (src_x1-src_x)*img[src_x0,src_y1,i] + (src_x-src_x0)*img[src_x1,src_y1,i]
                target_img[t_x,t_y,i] = int((src_y1-src_y)*r1+(src_y-src_y0)*r2)
    return target_img

img = cv.imread("./images/lenna.png")
d_img = billnear_func(img,(800,800))
cv.imshow("billnear", d_img)
cv.waitKey(0)



