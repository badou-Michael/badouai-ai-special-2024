import numpy as np
import cv2
from histogram import cv_imread
def bilinear_interpolation(filepath, dstH, dstW):
    # 读取图片
    srcimg = cv_imread(filepath)
    # 获取原始图片高，框，与通道
    srcH,srcW,channel = srcimg.shape
    dstimg = np.zeros((srcH,srcW,channel),dtype=np.uint8)
    # 缩放因子求取
    scale_x,scale_y = float(dstH/srcH),float(dstW/srcW)
    for i in range(channel):
        for dstx in range(dstH):
            for dsty in range(dstW):
                #坐标映射关系
                srcx = (dstx+0.5)*scale_x-0.5
                srcy = (dsty+0.5)*scale_y-0.5
                # 左上坐标横坐标，向下取整
                srcx0 = int(srcx)
                #右上坐标，如果边界超出了，则取边界值
                srcx1 = min(srcx0+1,srcW-1)
                #左下坐标
                srcy0 = int(srcy)
                #右下坐标
                srcy1 = min(srcy0+1,srcH-1)

                #各坐标像素值,
                p0,p1,p01,p02 = srcimg[srcx0,srcy0,i],srcimg[srcx1,srcy0,i],srcimg[srcx0,srcy1,i],srcimg[srcx1,srcy1]
                #沿srcx0，srcx1 一条线的点为H,这h的点的像素值为ph
                ph = (srcx-srcx0)*p1+(srcx1-srcx)*p0
                # 沿srcx0，沿srcx1 下边一条线的点为g,这h的点的像素值为pg
                pg = (srcx-srcx0)*p02 + (srcx1-srcx)*p01
                #根据pg,ph求，srcx,srcy的像素值
                dstimg[dstx,dsty,i] = (srcy1-srcy)*ph +(srcy-srcy0)*pg
    return dstimg

if __name__ == '__main__':
    dstimg = bilinear_interpolation("../第二周/lenna.png", 600.600)
    cv2.imshow("dstimg",dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()