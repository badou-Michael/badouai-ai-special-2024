import cv2
import numpy as np

def my_sxxczf(scr_image, new_size):
    scr_H, scr_W, channels = scr_image.shape  #获取原图像形状
    dst_w, dst_h =new_size
    if scr_H == dst_h and scr_W == dst_w:
        return scr_image.copy()
    scale_x = float(scr_W)/dst_w
    scale_y = float(scr_H)/dst_h
    #遍历目标图像，进行插值
    dst = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    for n in range(3):    # 对channel循环range(channel)；
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 目标在源上的坐标（浮点值）
                scr_x = (dst_x + 0.5)*scale_x - 0.5
                scr_y = (dst_y + 0.5)*scale_y - 0.5
                scr_x_0 = int(np.floor(scr_x))
                scr_y_0 = int(np.floor(scr_y))
                scr_x_1 = min(scr_x_0 + 1, scr_W - 1)
                scr_y_1 = min(scr_y_0 + 1, scr_H - 1)
                #
                val0 = (scr_x_1 - scr_x) * scr_image[scr_y_0, scr_x_0, n] + (scr_x - scr_x_0) * scr_image[scr_y_0, scr_x_1, n]
                val1 = (scr_x_1 - scr_x) * scr_image[scr_y_1, scr_x_0, n] + (scr_x - scr_x_0) * scr_image[scr_y_1, scr_x_1, n]
                dst[dst_y, dst_x, n] = int((scr_y_1 - scr_y) * val0 + (scr_y - scr_y_0) * val1)
    return dst

img_in = cv2.imread('lenna.png')

#img_out = cv2.resize(img_in, (800, 800))
img_out = my_sxxczf(img_in, (800, 800))

cv2.imshow('scr_image', img_in)
cv2.imshow('dst_image', img_out)
cv2.waitKey()

