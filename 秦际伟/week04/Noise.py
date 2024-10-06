import cv2
from skimage import util
import numpy as np
import random

img = cv2.imread('img/lenna.png')

# 高斯噪声 直接调用接口
gaussian_noise_image = util.random_noise(img,mode='gaussian')

#高斯噪声实现方法
def gaussina_noise(img ,means,sigma):
    img = img.copy()
    height, width, channels = img.shape
    guess_noise = np.zeros((height,width,channels), dtype=np.float64)
    for c in range(channels):
        for i in range(height):
            for j in range(width):
                guess_noise[i,j,c] = random.gauss(means, sigma)

    guess_noise_img = img + guess_noise
    guess_noise_img  = np.clip(guess_noise_img, 0, 255).astype(np.uint8)
    return guess_noise_img

gaussian_noise_image2 = gaussina_noise(img,2,16)


# 椒盐噪声
salt_noise_image = util.random_noise(img,mode='s&p', amount=0.05)

# 椒盐噪声实现方法
def p_salt_noise(img,salt_ratio):
    NoiseImg = img.copy()
    NoiseNum = int(salt_ratio * img.shape[0] * img.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)

        if random.random() <= 0.5:
            NoiseImg[randX, randY, :] = [0, 0, 0]
        else:
            NoiseImg[randX, randY, :] = [255, 255, 255]

    return NoiseImg

salt_noise_image2 = p_salt_noise(img,0.1)



# 泊松噪声
poisson_noise_image = util.random_noise(img,mode='poisson')


# 展示图像对比
cv2.imshow('lenna', img)
cv2.imshow('gaussian', gaussian_noise_image)
cv2.imshow('gaussian2', gaussian_noise_image2)
cv2.imshow('salt', salt_noise_image)
cv2.imshow('salt2', salt_noise_image2)
cv2.imshow('poisson', poisson_noise_image)
cv2.waitKey(0)
