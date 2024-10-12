"""
椒盐噪声——yzc
"""
import random
import cv2


def jyNoise(src,SNR):
    jyImg = src
    jyNum = int(SNR * jyImg.shape[0] *jyImg.shape[1])
    for i in range(jyNum):
        rangX = random.randint(0,jyImg.shape[0]-1)
        rangY = random.randint(0,jyImg.shape[1] -1)
        # jyImg[rangX,rangY] = jyImg[rangX,rangY]
        if jyImg[rangX,rangY] <= 0.5:
            jyImg[rangX,rangY] = 0
        else:
            jyImg[rangX,rangY] = 255
    return jyImg

img = cv2.imread("..\\lenna.png",0)
jyImg = jyNoise(img,0.5)
cv2.imshow("jyNoise", jyImg)
cv2.waitKey()
