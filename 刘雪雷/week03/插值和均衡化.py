import cv2
import numpy as np

def function(img):
    h, w, c = img.shape
    emptyimg = np.zeros((1000, 1000, c), img.dtype)
    htimes = 1000/h
    wtimes = 1000/w
    for i in range(1000):
        for j in range(1000):
            x = int(i/htimes+0.5)
            y = int(j/wtimes+0.5)
            emptyimg[i, j] = img[x, y]
    return emptyimg

img = cv2.imread('lenna.png')
result = function(img)
print(result)
cv2.imshow('zoom', result)
cv2.waitKey(0)

#直接调用接口 
result2 = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_NEAREST)
cv2.imshow('zoom2', result2)

#双向插值调用接口 

result3 = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_LINEAR)


#直方图均衡化
img = cv2.imread('lenna.png')
gray_img = rgb2gray(img)
#原图灰度直方图
histimg = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
#直方图均衡化
equimg = cv2.equalizeHist(gray_img)
#均衡化后的直方图
histimg2 = cv2.calcHist([equimg], [0], None, [256], [0, 256])

