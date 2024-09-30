##高斯采样分布公式, 得到输出像素Pout.    Pout = Pin + random.gauss
import cv2
import numpy as np
# from numpy import shape
import random
# src：原始图像。means：高斯噪声的均值。sigma：高斯噪声的标准差。percentage：表示要添加噪声的像素所占的百分比。
def GaussianNoise (src,means,sigma,percentage):
    Nosise_Img=src
    Noise_Num=int(percentage * src.shape[0]*src.shape[1])
    for i in range (Noise_Num) :#循环 NoiseNum 次，每次循环选择一个随机像素点。
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        #-1是因为索引从0开始
        randx=random.randint(0,src.shape[0]-1)
        randy= random.randint(0,src.shape[1]-1)
        ##添加高斯噪声,将原始像素值与高斯分布生成的随机数相加，生成噪声像素值。
        Nosise_Img[randx,randy]=Nosise_Img[randx,randy]+random.gauss(means,sigma)
        #重新将像素值放缩在[0~ 255]之间
        if Nosise_Img[randx,randy]<0:
            Nosise_Img[randx, randy]=0
        elif Nosise_Img[randx,randy]>255:
            Nosise_Img[randx, randy]=255
    return Nosise_Img

img=cv2.imread('lenna.png',0)##0为灰度，1是彩图
lenna_GaussianNoise=GaussianNoise(img,2,4,0.8)
##显示原图和噪音图像
cv2.imshow('src_gray',img)
cv2.imshow('lenna_GaussianNoise',lenna_GaussianNoise)
##保存高斯噪声结果图片
cv2.imwrite('lenna_GaussianNoise_cv2imwrite.png',lenna_GaussianNoise)##.png别忘了

##灰度2
img_gray2 = cv2.imread('lenna.png')
img2_gray = cv2.cvtColor(img_gray2,cv2.COLOR_BGR2GRAY)
cv2.imshow('source_2_gray',img2_gray)
cv2.waitKey(0)
