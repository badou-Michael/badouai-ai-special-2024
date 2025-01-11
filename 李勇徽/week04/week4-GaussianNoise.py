import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def GaussianNoise(src, mean, sigma, percentage):
    # get the total GaussianNoise number
    noiseImg = src
    NoiseNum = int(noiseImg.shape[0] * noiseImg.shape[1] * percentage)

    for i in range(NoiseNum):
        # get one random pixel each time
        noise_X = random.randint(0, noiseImg.shape[0] - 1)
        noise_Y = random.randint(0, noiseImg.shape[1] - 1)
        # add Gaussian Noise
        noiseImg[noise_X, noise_Y] += random.gauss(mean, sigma)
        # check whether the value is abnormal
        if noiseImg[noise_X, noise_Y] < 0:
            noiseImg[noise_X, noise_Y] = 0
        elif noiseImg[noise_X, noise_Y] > 255:
            noiseImg[noise_X, noise_Y] = 255
    return noiseImg

if __name__ == '__main__':
    img = cv2.imread('lenna.png',0)
    img_GN = GaussianNoise(img, 3, 5, 1)
    plt.subplot(1,2,1)
    plt.title('GaussianNoise Image')
    plt.imshow(img_GN, cmap = 'gray')
    plt.axis('off')

    img_org = cv2.imread('lenna.png', 0)
    plt.subplot(1,2,2)
    plt.title('Orginal Image')
    plt.imshow(img_org, cmap = 'gray')
    plt.axis('off')

    plt.show()
    # cv2.imshow('image_GaussianNoise', img_GN)
    # cv2.waitKey(0)
