import numpy as np
import cv2

# 定义双线性插值函数，参数：待处理图像和输出图像
def bilinear_interpolation(img,out_dim):
    # 获取源图像尺寸
    src_h, src_w, channel = img.shape
    # 从输出尺寸中获取目标图像的宽度和高度
    dst_h, dst_w = out_dim[1], out_dim[0]
    print ("src_h, src_w = ", src_h, src_w)
    print ("dst_h, dst_w = ", dst_h, dst_w)
    # 如果源图像和目标图像的尺寸相同，则直接返回源图像副本，无需进行插值
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    # 初始化一个与目标尺寸相同、通道数为3（RGB）的零矩阵，用于存储插值后的图像
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    # 计算x和y方向上的缩放比例
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    # 遍历像素点，计算其对应的源图像中的像素值
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):

                # 遍历图像的每个通道（RGB），然后是目标图像的每一行（dst_y）和每一列（dst_x）
                src_x = (dst_x + 0.5) * scale_x-0.5
                src_y = (dst_y + 0.5) * scale_y-0.5
 
                # 计算目标像素在源图像中的对应位置（使用几何中心对称的方法，以提高插值精度）
                src_x0 = int(np.floor(src_x))     #np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x1 = min(src_x0 + 1 ,src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
 
                # 确定源图像中与目标像素相邻的四个像素点的坐标
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                dst_img[dst_y,dst_x,i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
 
    return dst_img
 
 # 使用双线性插值公式计算目标像素的值，并将其存储在dst_img中
if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img,(700,700))
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey()
