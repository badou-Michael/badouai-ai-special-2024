# 实现双线性插值实现图像放大

import cv2
import numpy as np

def bilinear_interpolation(img,out_dim):
    src_h, src_w, src_c = img.shape
    dst_x, dst_y = out_dim[1], out_dim[0]
    dstp_img = np.zeros((dst_x, dst_y, 3), np.uint8)
    sh = float(src_h / dst_x)
    sw = float(src_w / dst_y)

    for dst_h in range(dst_x):
        for dst_w in range(dst_y):
            src_x=(dst_h+0.5)*sh-0.5
            src_y=(dst_w+0.5)*sw-0.5 # 实现几何中心对称

            src_x0=int(np.floor(src_x))
            src_x1=min(src_x0+1,src_h-1)
            src_y0=int(np.floor(src_y))
            src_y1=min(src_y0+1,src_w-1) # 找到目标图像在原图像映射之后点的四个邻近点/还加了边界条件判断

            # 两次x方向插值
            dst_R1=(src_x1-src_x)*img[src_x0,src_y0]+(src_x-src_x0)*img[src_x1,src_y0]
            dst_R2=(src_x1-src_x)*img[src_x0,src_y1]+(src_x-src_x0)*img[src_x1,src_y1]

            # 一次y方向插值
            dstp_img[dst_h,dst_w]=(src_y1-src_y)*dst_R1+(src_y-src_y0)*dst_R2

    return  dstp_img

if __name__ == '__main__':
    img=cv2.imread("lenna.png")
    dst=bilinear_interpolation(img,(700,700))
    cv2.imshow("bilinear_interpolation",dst)
    cv2.waitKey(0)







