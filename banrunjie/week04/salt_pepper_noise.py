import cv2
import numpy as np
import random

def salt_pepper_noise(img,percentage):
    noise_img = img.copy()
    flag_img = np.zeros_like(img,dtype=np.uint8)
    noise_num = int(img.shape[0]*img.shape[1]*percentage)
    while noise_num>0:
        x=random.randint(0,img.shape[0]-1)
        y=random.randint(0,img.shape[1]-1)
        if flag_img[x,y]==0:
            if random.random()>0.5:
                noise_img[x,y]=255
            else:
                noise_img[x,y]=0
            flag_img[x,y]=1
            noise_num-=1
        else:
            continue
    return noise_img

if __name__ == '__main__':
    im  = cv2.imread('lenna.png',cv2.IMREAD_GRAYSCALE)
    noise_im =salt_pepper_noise(im,0.5)
    cv2.imshow('imagine',im)
    cv2.imshow('noise imagine',noise_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
