"""
实现最邻近插值算法
"""

import cv2
import numpy as np

def nearest(img, nearest_height, nearest_width):
    #高，宽，通道
    height,width,channels = img.shape
    #三通道全零图片，一张黑色图
    emptyImage = np.zeros((nearest_height, nearest_width, channels), np.uint8)
    # cv2.imshow("emptyImage",emptyImage)
    # cv2.waitKey(0)

    #图片缩放的比值
    sh = nearest_height/height
    sw = nearest_width/width

    for i in range(nearest_height):
        for j in range(nearest_width):

            #int(),向下取整，保证取值都能在原图片内，需加0.5
            x=int(i/sh + 0.5)  
            y=int(j/sw + 0.5)
            emptyImage[i,j]=img[x,y]

    return emptyImage



img=cv2.imread("lenna.png")
nearest_height, nearest_width = 800, 800
zoom=nearest(img, nearest_height, nearest_width)

print(zoom)
print(zoom.shape)

cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)
