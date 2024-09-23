import cv2
import numpy as np

def nearest_intplt(img, method=1):
    # resize img to 800 x 800
    # input:  - img: the resized image
    #         - method: resize method
    # output: 0
    if method == 1:
        height, width, channel = img.shape
        img_intplt = np.zeros((800, 800, channel), np.uint8)
    
        sh = 800/height
        sw = 800/width
        
        for i in range(800):
            for j in range(800):
                x = int(i/sh + 0.5)
                y = int(j/sw + 0.5)
                img_intplt[i,j] = img[x,y]
                # print(img_intplt)
        img_zoom = img_intplt
    elif method == 2:
        img_zoom = cv2.resize(img, (800, 800), cv2.INTER_NEAREST)
    cv2.imshow('nearest interpolation', img_zoom)
    cv2.waitKey(0)
    return 0
    
if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    # resize image to 800 x 800
    # nearest_intplt - method1 : user define funcion
    # nearest_intplt - method2 : cv2.resize function
    img_zoom = nearest_intplt(img, 1)
    