# 双线性插值 这次将周围四个点的坐标和像素值对像素的影响都考虑进去了，对于行，将左右像素坐标和像素值都考虑进去了
# 还解决了一个问题， 缩放后坐标会存在偏移也就是映射点的位置不准确了，整体会偏移
import cv2
import numpy as np
# 双线性插值 原图像， 目标图像的尺寸 dimention 是尺寸的意思,destination”（目标、目的地）的缩写。
def bilinear_interpolution(img,out_dim):
    src_height,src_width,channels = img.shape
    dst_height,dst_width=out_dim[0],out_dim[1]
    if src_height ==dst_height and src_width==dst_width:
        return img.copy()   #直接输出显示img的复制图像
    #准备一个全为0 的图像矩阵
    dst_img = np.zeros((dst_height,dst_width,3),dtype=np.uint8)
    print(dst_height,dst_width,channels)
    #准备好缩放程度
    scale_x, scale_y =float(src_width)/dst_width, float(src_height)/dst_height
    for i in range(channels):
        for dst_y in range(dst_height):
            for dst_x in range(dst_width):
                #先实现位置的校准
                src_x = (dst_x+0.5)*scale_x-0.5
                src_y = (dst_y+0.5)*scale_y-0.5
                #在实现边缘的检测，由于要获取边缘四个点的像素值，目标x/y左边正好落在边缘，那会有bug,边缘没有值了，那就取边缘
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0+1, src_width-1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_height - 1)
                #print(f"Coordinates:dst_x={dst_x},src_x0={src_x0}, src_y0={src_y0}, src_x1={src_x1}, src_y1={src_y1}")
                #算法最后一步，核心一步看谁影响大
                temp0 = (src_x1 - src_x)*img[src_y0,src_x0,i]+(src_x - src_x0)*img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x)*img[src_y1,src_x0,i]+(src_x - src_x0)*img[src_y1,src_x1,i]
                dst_img[dst_y, dst_x, i]=int((src_y - src_y0)*temp1 + (src_y1 - src_y)*temp0)

    return dst_img

img=cv2.imread("lenna.png")
dst= bilinear_interpolution(img,(700,700))
cv2.imshow("source img",img)
cv2.imshow("bilinear_interpolution2",dst)
resized_img = cv2.resize(img, (700, 700))
cv2.imshow("resized_img",resized_img)
cv2.waitKey(0)
