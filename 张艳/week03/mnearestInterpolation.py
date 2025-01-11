import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

''' nearest interpolation '''

def nearest(img,w2,h2):
    w1,h1,c=img.shape
    w12=w1/w2
    h12=h1/h2
    imgNearest=np.zeros((w2,h2,c),np.uint8)
    for i in range(w2):
        for j in range(h2):
            x=int(i*w12 + 0.5) #加0.5再取整，是为了找最邻近的点
            y=int(j*h12 + 0.5)
            imgNearest[i,j]=img[x,y]
    return imgNearest

img = cv2.imread("lenna.png") #512,512,3 #plt.imread()有问题！,,,cv2.imread()
imgNearest=nearest(img,300,600)
print(img.shape)
print(imgNearest.shape)
cv2.imshow("imgNearest", imgNearest)
cv2.imshow("img", img)
cv2.waitKey(0)

# plt.subplot(2,1,1)
# plt.imshow(img)
# plt.subplot(2,1,2)
# plt.imshow(imgNearest)
# plt.show()
