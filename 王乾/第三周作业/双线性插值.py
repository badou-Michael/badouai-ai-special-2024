import cv2
import numpy as np


def interpolation(image,dst):
    h,w,t = image.shape
    dst_h,dst_w = dst[1],dst[0]
    print("src_h, src_w = ", h, w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if h==dst_h and w ==dst_w:
        return image.copy()
    new_image = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    bili_x = float(w)/dst_w
    bili_y = float(h)/dst_h
    for i in range(t):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                #这一步是求拓展图基于原图的中心坐标
                scr_x = (dst_x+0.5)*bili_x-0.5
                scr_y = (dst_y+0.5)*bili_y-0.5

                #这一步是求出P点的上下左右的整数做标点
                scr_x0 = int(np.floor(scr_x))
                scr_x1 = min(scr_x0+1,w-1)
                scr_y0 = int(np.floor(scr_y))
                scr_y1 = min(scr_y0+1,h-1)

                #这一步是求出点P四个角的坐标像像素值
                temp0 = (scr_x1-scr_x)*image[scr_y0,scr_x0,i]+(scr_x-scr_x0)*image[scr_y0,scr_x1,i]
                temp1 = (scr_x1-scr_x)*image[scr_y1,scr_x0,i]+(scr_x-scr_x0)*image[scr_y1,scr_x1,i]
                new_image[dst_y,dst_x,i] = int((scr_y1-scr_y)*temp0+(scr_y-scr_y0)*temp1)
    return  new_image

img = cv2.imread("lenna.png")
cv2.imshow("窗口1",interpolation(img,(1000,1000)))
cv2.waitKey()
