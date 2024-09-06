import cv2 as cv
import numpy as np
image = cv.imread("lenna.png")
h = np.shape(image)[0]
w = np.shape(image)[1]
grayimg = np.zeros((h,w,3),np.uint8) #np.uint8取值范围0-255，常用来表示像素的灰度值；
for i in range(h):
    for j in range(w):
        grayimg[i,j] = int(image[i,j][0]*0.11+image[i,j][1]*0.59+image[i,j][2]*0.3)
cv.imshow("1",image)
cv.imshow("2",grayimg)
cv.waitKey(0)

#二值化
#简单阈值是选取一个全局阈值，灰度图大于阈值就赋值为255，否则为0；
import matplotlib.pyplot as plt
ret,mask_all = cv.threshold(src=grayimg,
                            thresh=127,
                            maxval=255,
                            type=cv.THRESH_BINARY)
print("全局阈值的shape", mask_all.shape)
plt.subplot(221)
plt.imshow(mask_all,cmap='gray')
plt.show()
plt.title("全局阈值")

#自适应阈值
grayimg1 = cv.imread("lenna.png",cv.IMREAD_GRAYSCALE)
mask_local=cv.adaptiveThreshold(src=grayimg1,
                                maxValue=255,
                                adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
                                thresholdType=cv.THRESH_BINARY,
                                blockSize=11,
                                C=2)
print("局部阈值的shape", mask_local.shape)
plt.subplot(222)
plt.imshow(mask_local,cmap='gray')
plt.show()
plt.title("jubu阈值")
