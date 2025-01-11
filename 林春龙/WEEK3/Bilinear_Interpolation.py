"""
实现图片双线性插值
和中心重合
"""

import numpy as np
import cv2


def bilinear_interpolation(img, out_dim):
    #原图
    src_h, src_w, channel = img.shape
    #生成图高，宽
    dst_h, dst_w = out_dim[1], out_dim[0]
    #尺寸不变
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    
    #三通道全零图片，一张黑色图
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)

    #计算图片的缩放比例
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h

    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
 
                #实现原图和输出图中心点的重合
                #并找到输出图在原图上对应的像素坐标x和y
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
 
                #根据原图像对应的像素坐标进行插值计算
                #scr_x0\scr_x1\scr_y0\scr_y1,两两组合得到邻近的4个坐标值
                src_x0 = int(np.floor(src_x))     #np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x1 = min(src_x0 + 1, src_w - 1)  #由于都是邻近坐标，保证不越界
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
 
                #计算插值点，代入公式（邻近坐标，分母都为1），先计算原图图像所在x的直线，在计算出直线x上的坐标
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                dst_img[dst_y,dst_x,i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
 
    return dst_img


if __name__ == "__main__":
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, (700, 700))
    #图片更加光滑
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()
