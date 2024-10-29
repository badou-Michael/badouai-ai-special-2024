'''
@Project ：BadouCV 
@File    ：salt_noise.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/10/15 00:01 
'''

import random
import cv2

def to_saltNoise (src_img,noiseLevel):
    '''
    实现salt noise 椒盐噪声
    :param src_img: 原图
    :param noiseLevel: 噪声等级
    :return: img
    '''

    img = src_img.copy()
    h, w = img.shape[:2]
    noiseNum = int(noiseLevel * h * w)  # 噪声像素个数

    for i in range(noiseNum):
        x = random.randint(0,w-1)
        y = random.randint(0,h-1)
        #random.random() 生成一个0到1之间的随机小数（包括0，但不包括1）。
        if random.random()<=0.5:
            img[x, y] = 0
        else:
            img[x,y]=255
    return img

img=cv2.imread('../lenna.png')
noise_img=to_saltNoise(img,0.5)
noise_img_gray=to_saltNoise(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),0.5)

cv2.imshow('src',img)
cv2.imshow('noise',noise_img)
cv2.imshow('noise2',noise_img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
