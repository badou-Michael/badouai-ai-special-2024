import numpy as np
import cv2
import random

def PepperSaltNoise(src, percetage):
    noise_img = src.copy()

    # 获取需要增加椒盐噪声的像素点数量
    noise_num = int(percetage * src.shape[0] * src.shape[1])

    # 循环给修改像素值
    for i in range(noise_num):
        # randX 随机生成的行 randY随机生成的列
        randX = random.randint(0, src.shape[0] - 1)  # random.randint 生成随机整数
        randY = random.randint(0, src.shape[1] - 1)

        # random.random 生成随机浮点数，介于0-1之间。 随机取值
        if random.random() <= 0.5:
            noise_img[randX,randY] = 0
        else:
            noise_img[randX,randY] = 255

    return noise_img

# 效果展示
img = cv2.imread('lenna.png',0)

noise_img = PepperSaltNoise(img,0.8)

combined_img = np.hstack([img,noise_img])
cv2.imshow('combined_img',combined_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
