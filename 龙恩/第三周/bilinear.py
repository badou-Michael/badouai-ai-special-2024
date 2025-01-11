import numpy as np
import cv2

def bilinear (img, dim):
    img=cv2.imread(img)
    in_w,in_h,in_c=img.shape
    out_w,out_h=dim[0],dim[1]
    if out_w==in_w and out_h==in_h:
        return img.copy()
    out_img=np.zeros((out_w,out_h,in_c),dtype=np.uint8)
    factor_w,factor_h=float(in_w)/out_w,float(in_h)/out_h
    for i in range(dim[0]):
        for j in range(dim[1]):
            #center point matching,find knowed outcome point from unknowed original point
            x=(i+0.5)*factor_w-0.5
            y=(j+0.5)*factor_h-0.5
            #find the four points
            x1=int(np.floor(x))
            x2=min(x1+1,in_w-1)
            y1=int(np.floor(y))
            y2=min(y1+1,in_h-1)
            #calculation
            Q11=img[x1,y1]
            Q12=img[x1,y2]
            Q21=img[x2,y1]
            Q22=img[x2,y2]
            
            R1=(x2-x)*Q11+(x-x1)*Q21
            R2=(x2-x)*Q12+(x-x1)*Q22
            out_img[i,j]=((y2-y)*R1)+((y-y1)*R2)
            
    cv2.imshow("Bilinear image",out_img)
    cv2.waitKey(0)

    return 

if __name__ == '__main__':
    bilinear("lenna.png",(700,700))


            
                   
