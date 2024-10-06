import cv2
import numpy as np
def large(img):
    h,w,channel=img.shape
    output_img=np.zeros((900,900,channel),np.uint8)
    sw=900/w
    sh=900/h
    for i in range(900):
        for j in range(900):
            x=int(i/sw+0.5)
            y=int(j/sh+0.5)
            output_img[i,j]=img[x,y]
    return output_img
def main():
    img=cv2.imread("lenna.png")
    output=large(img)
    cv2.imshow("before",img)
    cv2.imshow("after",output)
    cv2.waitKey(0)
if __name__=="__main__":
    main()
    
