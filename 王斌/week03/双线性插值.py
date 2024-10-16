
import cv2
import numpy as np

def function(img):
    height, width, channel = img.shape
    imageInterp = np.zeros((800,800,channel),np.uint8)
    h = height/800
    w = width/800
    for i in range(800):
        for j in range(800):
            for z in range(channel):
                px = (i+0.5)*h-0.5
                py = (j+0.5)*w-0.5

                x0 = int(np.floor(px))
                x1 = min(x0+1,height-1)
                y0 = int(np.floor(py))
                y1 = min(y0+1,width-1)

                n1 = (x1-px)*img[x0, y0, z]+(px-x0)*img[x1, y0, z]
                n2 = (x1-px)*img[x0, y1, z]+(px-x0)*img[x1,y1, z]

                imageInterp[i, j, z] = (y1-py)*n1+(py-y0)*n2
    return imageInterp
img = cv2.imread("C:/Users/Administrator/Desktop/123.jpg")
zoom=function(img)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)
