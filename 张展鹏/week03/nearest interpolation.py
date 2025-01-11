import matplotlib.pyplot as plt
import cv2
import numpy as np

#1.定义一个函数处理最邻近插值
def nearest_img(img):
    h,w,n = img.shape #读取图片长宽通道数
    enlargeImg = np.zeros((1000,1000,n),dtype=np.uint8)  #创建一个1000*1000像素的n通道0矩阵图
    kh = 1000/h #计算缩放比例
    kw = 1000/w
    for goalH in range(1000):
        for goalW in range(1000):
            enlargeH = int(goalH/kh + 0.5) #int(),转为整型，使用向下取整。
            enlargeW = int(goalW/kw + 0.5) #加0.5的作用在于，让向下取整更贴近于四舍五入，比如2.5向下取整为2，四舍五入却为3，加0.5以后，即可弥补。
            enlargeImg[goalH,goalW] = img[enlargeH,enlargeW]
    return enlargeImg




if __name__ == '__main__':
    #调用函数实现
    sourceImg = cv2.imread('../lenna.png')
    h,w,n = sourceImg.shape
    enlargeImg = nearest_img(sourceImg)
    cv2.imshow('source_img',sourceImg)
    cv2.imshow('nearest_img',enlargeImg)
    #直接使用cv2函数实现 （interpolation=cv2.INTER_NEAREST 最邻近插值）    （interpolation=cv2.INTER_LINEAR 双线性插值）
    # enlargeImg3 = cv2.resize(sourceImg,(1000, 1000),interpolation=cv2.INTER_NEAREST) #最邻近插值比较简单粗暴
    # enlargeImg2 = cv2.resize(sourceImg,(1000,1000),interpolation=cv2.INTER_LINEAR) #双线性插值图片更加清晰，但算法更加复杂
    # cv2.imshow('111',enlargeImg2)
    # cv2.imshow('222', enlargeImg3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
