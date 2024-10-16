#实现高斯噪声，椒盐噪声
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

#实现高斯噪声
def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        for c in range(src.shape[2]):
            NoiseImg[randX,randY,c]+=random.gauss(means,sigma)
            if NoiseImg[randX,randY,c]<0:
                NoiseImg[randX,randY,c]=0
            elif NoiseImg[randX,randY,c]>255:
                NoiseImg[randX,randY,c]=255

    return NoiseImg

#实现椒盐噪声
def fun(src,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        for c in range(src.shape[2]):
            if random.random()<=0.5:
                NoiseImg[randX,randY]=0
            else:
                NoiseImg[randX,randY]=255

    return NoiseImg

if __name__ == '__main__':
    photo=cv2.imread("ho.jpg")
    cv2.imshow("yuantu", photo)
    # 高斯噪声
    img1=GaussianNoise(photo,2,4,1.5)
    cv2.imshow("gaosi",img1)

    #椒盐噪声
    img3=fun(photo,0.8)
    cv2.imshow("jiaoyan",img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
