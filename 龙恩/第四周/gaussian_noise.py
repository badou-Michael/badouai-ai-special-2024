import numpy as np
import cv2
import random
from numpy import shape

def gaussian(path,mean,sigma,percent):
    img_in=cv2.imread(path,0)
    img_out=img_in.copy()
    number=int(percent*img_out.shape[0]*img_out.shape[1])
    #pick a point
    for i in range(number):
        x=random.randint(0,img_out.shape[0]-1)
        y=random.randint(0,img_out.shape[1]-1)
        img_out[x,y]=np.clip(img_in[x,y]+random.gauss(mean,sigma),0,255)
    cv2.imshow('original',img_in)
    cv2.imshow('gaussian',img_out)
    cv2.waitKey(0)


gaussian("lenna.png",2,4,0.8)

