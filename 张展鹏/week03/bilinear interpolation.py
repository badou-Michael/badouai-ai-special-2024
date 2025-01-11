import matplotlib.pyplot as plt
import cv2
import numpy as np


def billnear_img(img,goal):
    h,w,n = img.shape
    goalH,goalW = goal[1],goal[0]
    if h == goalH and w == goalW: #如果操作后的图片像素大小与原图相同，则直接复制
        return img.copy()
    sh,sw = float(goalH) / h ,float(goalW) / w #计算缩放比例
    goalImg = np.zeros((goalH,goalW,3),dtype=np.uint8) #创建一个目标全0矩阵
    for i in range(n):
        for y in range(goalH):
            for x in range(goalW):
                srcx = (x + 0.5) / sw - 0.5 #原图与目标图中心对称
                srcy = (y + 0.5) / sh - 0.5

                srcx0 = int(np.floor(srcx)) ##np.floor()返回不大于输入参数的最大整数。（向下取整）
                srcy0 = int(np.floor(srcy))
                srcx1 = min(srcx0 + 1 ,w - 1) #边界值判断，如果算出来的值小于边界值，则一直取前面，如果当算到边界点时，srcx0+1超出边界，所以取 w-1 因为索引从0开始所以边界为w-1
                srcy1 = min(srcy0 + 1 ,h - 1)

                temp0 = (srcx1 - srcx) * img[srcy0,srcx0,i] + (srcx - srcx0) * img[srcy0,srcx1,i] #根据公式计算两次x，一次y
                temp1 = (srcx1 - srcx) * img[srcy1,srcx0,i] + (srcx - srcx0) * img[srcy1,srcx1,i]
                goalImg[y,x,i] = int((srcy1 - srcy) * temp0 + (srcy - srcy0) * temp1)
    return goalImg

if __name__ == '__main__':
    sourceImg = cv2.imread('../lenna.png')
    # enlargeImg2 = cv2.resize(sourceImg,(1000,1000),interpolation=cv2.INTER_LINEAR) #可使用cv2函数直接双线性插值
    goalImg = billnear_img(sourceImg,(1000,1000))
    cv2.imshow('sourceImg',sourceImg)
    cv2.imshow('goalImg',goalImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
