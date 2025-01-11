import numpy as np
import cv2
import random

def salt_pepper (src,percentage):
    Noise_Image=src
    Noise_Num=int(percentage * src.shape[0] * src.shape[1] )
    for i in range(Noise_Num): #循环遍历每个需要添加噪声的像素点。
        randx = random.randint(0, src.shape[0] - 1)
        randy = random.randint(0, src.shape[1] - 1)
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        #使用random.random生成一个随机浮点数，以决定该像素点是变为黑色（0）还是白色（255）。
        if random.random()<=0.5 :
            Noise_Image[randx,randy]=0
        else:
            Noise_Image[randx,randy]=255

    return Noise_Image


img=cv2.imread('lenna.png',0) #使用cv2.imread读取名为lenna.png的图像，并将其转换为灰度图像。
cv2.imshow('lenna.png_gray',img)
img_salt_pepper=salt_pepper(img,0.8)
#保存椒盐噪声图片
##查看
cv2.imshow('lenna_PepperandSalt',img_salt_pepper)
##保存
# cv2.imwrite('lenna_PepperandSalt.png',img_salt_pepper)


###灰度 2
img=cv2.imread('lenna.png')
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('lenna_gray2',img_gray)
cv2.waitKey(0)


