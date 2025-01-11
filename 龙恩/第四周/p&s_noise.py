import numpy as np
import cv2
import random
from numpy import shape


def peppersalt(path,percent):
    img_in=cv2.imread(path,0)
    img_out=img_in.copy()
    number=int(percent*img_in.shape[0]*img_in.shape[1])
    #pick a point
    for i in range(number):
        x=random.randint(0,img_in.shape[0]-1)
        y=random.randint(0,img_in.shape[1]-1)
        img_out[x,y]=0 if random.random()<0.5 else 255
    cv2.imshow("original",img_in)
    cv2.imshow("p&s",img_out)
    cv2.waitKey(0)

peppersalt("lenna.png",0.2)


    

'''
for not repeat:

def peppersalt(path,percent):
    img_in=cv2.imread(path,0)
    img_out=img_in.copy()
    number=int(percent*img_in.shape[0]*img_in.shape[1])
    #pick a point
    pixelset=set()
    for i in range(number):
        while True:
            x=random.randint(0,img_in.shape[0]-1)
            y=random.randint(0,img_in.shape[1]-1)
            if (x,y) not in pixelset:
                pixelset.add((x,y))
                img_out[x,y]=0 if random.random()<0.5 else 255
                break
    cv2.imshow("original",img_in)
    cv2.imshow("p&s",img_out)
    cv2.waitKey(0)

peppersalt("lenna.png",0.2)
'''
