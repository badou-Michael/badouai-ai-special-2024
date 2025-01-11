import numpy as np
import cv2
# w:stand for the height of the outcome image
# h: stand for the width of the outcome image
def nearest(img,out_w,out_h):
    img=cv2.imread(img)
    in_w,in_h,in_c=img.shape
    out_img=np.zeros((out_w,out_h,in_c),np.uint8)
    factor_w,factor_h=out_w/in_w,out_h/in_h
    for i in range(out_w):
        for j in range(out_h):
            x=int(i/factor_w+0.5)
            y=int(j/factor_h+0.5)
            out_img[i,j]=img[x,y]
    cv2.imshow("original",img)
    cv2.imshow("nearest",out_img)
    cv2.waitKey(0)
    return 


nearest("lenna.png",800,800)
    
    
