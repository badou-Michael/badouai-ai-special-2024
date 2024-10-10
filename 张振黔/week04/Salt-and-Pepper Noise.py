import numpy as np
import cv2
import random
def Noice(src,percentage):
    NoiceImg=src.copy() #防止修改原图
    NoiceNum=int(percentage*src.shape[0]*src.shape[1]) #按百分比生成椒盐噪声
    for i in range(NoiceNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        if random.random()<0.5:
            NoiceImg[randX,randY]=0
        else:
            NoiceImg[randX,randY]=255
    return NoiceImg
img=cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Noiceimage=Noice(gray,0.1)
imgs=np.hstack((gray,Noiceimage))
cv2.imshow('compare',imgs)
cv2.waitKey()
