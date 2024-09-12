import cv2
import numpy as np
def BL_interpolation(image,n):
    src_h,src_w,src_c=image.shape
    dst_h,dst_w=src_h*n,src_w*n
    if n==1: #缩放比例为1时返回原图
        return image.copy()
    emptyImage=np.zeros((dst_h,dst_w,src_c),np.uint8)
    for i in range(src_c):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                #对齐几何中心,进一步缩放缩放图 x y 值
                src_x=(dst_x+0.5)/n-0.5
                src_y=(dst_y+0.5)/n-0.5
                #找到对应的四个插值原图点,赋上channel值
                src_x0=int(min(src_x,src_w-1))
                src_x1=int(min(src_x+1,src_w-1))#边界处理，判断右上点索引是否超出原图边界W-1
                src_y0=int(min(src_y,src_h-1))
                src_y1=int(min(src_y+1,src_h-1))
                #计算channel值
                src_c0 = image[src_y0, src_x0, i]#四个节点对应四个channel值;y对应行（高度），x对应列（宽度）
                src_c1 = image[src_y1, src_x0, i]
                src_c2 = image[src_y0, src_x1, i]
                src_c3 = image[src_y1, src_x1, i]
                delta_x=src_x-src_x0
                delta_y=src_y-src_y0
                #两次x插值
                dst_cx1 = int(src_c1 * delta_x + src_c0 * (1 - delta_x))
                dst_cx2 = int(src_c3 * delta_x + src_c2 * (1 - delta_x))
                #y插值
                dst_c = int(dst_cx1 * delta_y + dst_cx2 * (1 - delta_y))
                #将channel值赋给输出矩阵
                emptyImage[dst_y,dst_x,i]=dst_c
    return emptyImage

image=cv2.imread('lenna.png')
scalefactor=2
zoom_BL=BL_interpolation(image,scalefactor)
cv2.imshow('source',image)
cv2.imshow('zoom',zoom_BL)
print('---source image---\n',image)
print('---zoom image---\n',zoom_BL)
cv2.waitKey()
