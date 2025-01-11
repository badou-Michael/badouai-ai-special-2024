#!/usr/bin/python
# -*- coding: utf-8 -*-
 
import numpy as np
import cv2
 
'''
python implementation of bilinear interpolation
'''
#双线性插值，四个点都考虑，并且按照距离分配对应去权重，具体就是先考虑横向，在考虑纵向
def bilinear_interpolation(img,out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0] #分辨率： 宽*高
    print ("src_h, src_w = ", src_h, src_w)
    print ("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8) #3：三通道的意思，np.uint8： 无符号整数
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h #原来的比现在的，是个小于1的数
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # dst_y: 固定了某一行，也就是纵坐标y
                # dst_x: 固定了某一列，也就是横坐标x
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                #保证放大后和原图的几何中心对齐（这样找到的四个点才是合适的），两边src 和 dst同时 + 0.5 就可以，自己证明
                src_x = (dst_x + 0.5) * scale_x-0.5
                src_y = (dst_y + 0.5) * scale_y-0.5

                # find the coordinates of the points which will be used to compute the interpolation
                src_x0 = int(np.floor(src_x))     #np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x1 = min(src_x0 + 1 ,src_w - 1) # 防止+1后越界
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)  # (src_x0,src_y0):左下，(src_x0,src_y0)：右上
 
                # calculate the interpolation
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                dst_img[dst_y,dst_x,i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
 
    return dst_img
 
 
if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img,(700,700))  # 由 512*512 扩展到 700*700
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey()
