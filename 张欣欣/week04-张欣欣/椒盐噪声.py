import cv2
import random
from numpy import shape

def PepperSoltNoice(src,snr):
    PepperSoltImg=src
    PepperSoltNum=int(snr*PepperSoltImg.shape[0]*PepperSoltImg.shape[1])
    for i in range(PepperSoltNum):
        randX=random.randint(0,PepperSoltImg.shape[0]-1)
        randY=random.randint(0,PepperSoltImg.shape[1]-1)
        if random.random()<=0.5:
            PepperSoltImg[randX,randY]=0
        else:
            PepperSoltImg[randX,randY]=255
    return PepperSoltImg


img=cv2.imread("lenna.png")
img1=PepperSoltNoice(img,0.6)
img=cv2.imread("lenna.png")
cv2.imshow("Img",img)
cv2.imshow("PepperSoltImg",img1)
cv2.waitKey(0)
