'''
@Project ：BadouCV 
@File    ：_intsertValue.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/10/10 15:34 
'''
import cv2
import numpy as np
img=cv2.imread('../lenna.png')
def nearestValueInsert (img,out_pixel):
    h,w,chanels=img.shape
    img0=np.zeros((out_pixel[0],out_pixel[1],chanels),np.uint8)
    h_rate=out_pixel[0]/h
    w_rate=out_pixel[1]/w
    for j in range(out_pixel[1]):
        for i in range(out_pixel[0]):
            x = int(i/w_rate+0.5) # int 向下取整， 所以加0.5 模拟四舍五入
            y = int(j/h_rate+0.5)
            img0[i,j]=img[x,y]
    return img0

zoom=nearestValueInsert(img,(800,800))
print(zoom)
cv2.imshow('neInsert',zoom)
cv2.waitKey(0)
cv2.destroyAllWindows()
