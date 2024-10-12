"""
插值，yzc
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np


# 1.最临近插值 --计算过程

def function(img):
    h,w,c = img.shape
    kongImg = np.zeros((800,800,c),np.uint8)
    suoh = 800/h
    suow = 800/w
    for i in range(800):
        for j in range(800):
            x = int(i/suoh + 0.5)
            y = int(j/suow + 0.5)
            kongImg[i,j] = img[x,y]
    return  kongImg

img = cv2.imread("..\\lenna111.png")
cv2.imshow("yuantu img:",img)
imgCV = function(img)
cv2.imshow("yzc img near tuidao",imgCV)

# cv2.waitKey()
# # 通过plt画布展示
# img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# newImg = function(img1)
# plt.subplot(111)
# plt.imshow(newImg)
# plt.show()
####------------------------------------------------####

####2.最邻近插值  --resize
imgResize = cv2.resize(img,(600,600),interpolation=cv2.INTER_NEAREST)
cv2.imshow("near resize img:",imgResize)
cv2.waitKey()
