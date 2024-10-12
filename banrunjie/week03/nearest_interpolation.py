import cv2
import numpy as np

def nearest_interpolation(image):
    height, width , channels = image.shape
    new_image = np.zeros((700,700,channels),dtype=np.uint8)
    rh,rw=height/700,width/700
    for i in range(700):
        for j in range(700):
            x,y=int(i*rh+0.5),int(j*rw+0.5)
            new_image[i,j]=image[x,y]
    return new_image

if __name__ == '__main__':
    im = cv2.imread('lenna.png')
    new_im = nearest_interpolation(im)
    cv2.imshow('image',im)
    cv2.imshow('new image',new_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


