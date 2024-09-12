"""
实现图片灰度化
实现图片二值化
"""

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


#实现图片的灰度化和二值化
def toGrayBinary(img):
    h,w = img.shape[:2]                                    #获取图片的高和宽   
    img_gray = np.zeros([h,w], img.dtype)                  #创建与原来图片一样大小的单通道图片
    test = img[0,0]
    print(test)
    
    for i in range(h):
        for j in range(w):
            m = img[i,j]                                   #取出当前高和宽中的 BGR  坐标
            img_gray[i,j] = int(m[0]*0.1 + m[1]*0.6 + m[2]*0.3)
    
    print(img_gray)
    print("image show gray: %s"%img_gray)
    cv2.imshow("image show gray",img_gray)

    plt.subplot(221)
    img = plt.imread("lenna.png") 
    # img = cv2.imread("lenna.png", False) 
    plt.imshow(img)
    print("---image lenna----")
    print(img)

    # 图片灰度化
    img_gray = rgb2gray(img)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray = img
    plt.subplot(222)
    plt.imshow(img_gray, cmap='gray')
    print("---image gray----")
    print(img_gray)

    # 图片二值化
    img_binary = np.where(img_gray >= 0.5, 1, 0) 
    print("-----imge_binary------")
    print(img_binary)
    print(img_binary.shape)

    plt.subplot(223) 
    plt.imshow(img_binary, cmap='gray')
    plt.show()

img = cv2.imread("lenna.png")
toGrayBinary(img)
