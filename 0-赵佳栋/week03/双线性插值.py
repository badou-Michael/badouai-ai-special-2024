'''
@Project ：BadouCV 
@File    ：bilinear_insertValue.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/10/10 16:28 
'''

import cv2
import numpy as np

# 方法一： 先求出目标图像（x，y）映射到原图的坐标（src_x，src_y），然后求出（src_x，src_y）这个点周围的四个坐标（x0，y0),(x0，y1），（x1，y0），（x1，y1）
# 以两次单线性叠加的公式带入（x0，y0),(x0，y1），（x1，y0），（x1，y1）四个点计算出x方向的插值p1，p2，最终根据p1，p2求出y方向的插值p

def bn_insert (img,out_pixel):
    h0, w0, chanels = img.shape
    # 根据目标分辨率新建一个空白图像
    h1 = out_pixel[0]
    w1 = out_pixel[1]
    img0=np.zeros((h1,w1,3),img.dtype)

    # 映射比例
    # h_rate = float (h0)/h1
    # w_rate = float (w0)/w1
    h_rate = h0 / h1
    w_rate = w0 / w1


    for i in range(chanels):
        for dst_y in range(h1):
            for dst_x in range (w1):

                # 还原目标图像坐标点在原图中映射的坐标src_x src_y, 使用几何中心对称的映射
                src_x = (dst_x+0.5)*w_rate - 0.5
                src_y = (dst_y+0.5)*h_rate - 0.5

                # src_x向下取整表示x0  x0+1表示x1  ：  x0---x-------x1    y同理
                src_x0 = int(src_x)
                src_x1 = min(src_x0+1,w0- 1) # 沿着x方向计算时，计算到边界的时候，src_x0+1会超出边界1个像素，则src_x1取边界值   y方向同理
                src_y0 = int(src_y)
                src_y1 = min(src_y0+1,h0-1)



                # X方向上的两次单线性插值得到的两个点
                p1 = img[src_x0,src_y0, i ]*(src_x1 - src_x) + img[src_x1,src_y0, i ]*(src_x - src_x0)
                p2 = img[src_x0,src_y1, i ]*(src_x1 - src_x) + img[src_x1,src_y1 , i ]*(src_x - src_x0)

                # 根据得到的两个点 计算y方向的单线性插值
                p = int(p1 * (src_y1 - src_y) + p2 * (src_y - src_y0 ))
                img0[dst_x,dst_y,i] = p


    print('原图高=',h0,'目标图高=',h1,'原图宽=',w0,'目标图宽=',w1)

    return img0




# ==============方法二  用权重思维  进行双线性插值================================================================
# 即 根据映射点与其周围四个点之间距离的权重计算插值

def bn_insert2(img,out_pixel,corner_align=False):
    w0,h0,chanels = img.shape
    w1,h1 = out_pixel[0],out_pixel[1]
    img0 = np.zeros((w1,h1,chanels),img.dtype)

    # 用角对齐（corner=True ）的方式  x轴 y轴的缩放比率,就是像素点的坐标数直接比 ，像素点坐标数=实际分辨率-1 如3x3图像最后一个点是 （2，2）
    scale_x_corner = float(w0-1)/(w1-1)
    scale_y_corner = float(h0-1)/(h1-1)

    #用边对齐方式，x轴和y轴的缩放比率就是原图和目标图 的 宽和高直接相比
    scale_x = float(w0)/(w1)
    scale_y = float(h0)/(h1)

    for i in range(chanels):
        for out_y in range(h1):
            for out_x in range(w1):
                if corner_align==True:
                    # 以四角对齐方式计算像素坐标（角对齐）
                    src_x = out_x * scale_x_corner
                    src_y = out_y * scale_y_corner
                else:
                    # 以边对齐方式计算像素坐标（边对齐）
                    src_x = (out_x + 0.5) * scale_x - 0.5
                    src_y = (out_y + 0.5) * scale_y - 0.5




                src_x0 = int(src_x)
                src_x1 = min(src_x0+1, w0-1) # 沿着x方向计算时，计算到边界的时候，src_x0+1会超出边界1个像素，则src_x1取边界值   y方向同理
                src_y0 = int(src_y)
                src_y1 = min(src_y0+1, h0-1)


                xd=src_x - src_x0
                yd=src_y - src_y0
                # 计算出映射点(src_x,src_y)周围四个点的像素值
                p00 = img[src_x0,src_y0]
                p01 = img[src_x0,src_y1]
                p10 = img[src_x1,src_y0]
                p11 = img[src_x1,src_y1]


                p1 = xd * p10 + (1-xd) * p00
                p2 = xd * p11 + (1-xd) * p01
                p =  yd * p2 + (1-yd) * p1

                img0[out_x,out_y] = p

    return img0








if __name__ == '__main__':
    img=cv2.imread('../lenna.png')

    # 用常规方式
    # out_img=bn_insert(img,(800,800))
    # cv2.imshow('bilinear_insertValue_img', out_img)

    # 用加权方式计算 （corner_align=False通过边对齐方式 / corner_align==True表示使用角对齐方式）
    out_img=bn_insert2(img,(800,800),True)

    cv2.imshow('bilinear_insertValue  2_img',out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
