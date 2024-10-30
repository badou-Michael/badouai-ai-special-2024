import cv2
import matplotlib.pyplot as plt
import numpy as np

def method(img, new_h, new_w):
    h, w, c = img.shape
    Tar_img = np.zeros((new_h,new_w,c),np.uint8)
    s_h, s_w = h/new_h, w/new_w
    for i in range(new_h):
        for j in range(new_w):
            x = min(int(i*s_h + 0.5),h - 1)
            y = min(int(j*s_w + 0.5),w - 1)
            Tar_img[i,j] = img[x,y]
    return Tar_img


if __name__ == '__main__':
    img = cv2.imread('girls.jpg')

    cv2.imshow('img',img)


    near_img = method(img,600,600)
    cv2.imshow('near_img',near_img)
    cv2.waitKey(0)
