import numpy as np
import cv2
from histogram import cv_imread

# 最近邻插值，输入图片路劲，图片的宽和高
def nearest_neighbor_interpolation(filepath,tarH,tarW):
    oriimg = cv_imread(filepath)
    # 获取原图像高，宽，通道
    oriH,oriW,channel = oriimg.shape
    # 创建空白图片矩阵
    tarimg = np.zeros((tarH,tarW,channel),dtype=np.uint8)
    # 获取缩放比例
    scale_x,scale_y = oriH/tarH,float(oriW)/tarW
    for i in range(channel):
        for m in range(tarH):
            for n in range(tarW):
                 orix = int(m*scale_x+0.5)
                 oriy = int(n*scale_y+0.5)
                 tarimg[m,n,i] = oriimg[orix,oriy,i]
    return tarimg

if __name__ == '__main__':
    dstimg = nearest_neighbor_interpolation("../第二周/lenna.png", 700, 700)
    cv2.imshow("dstimg",dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()