import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def PepperandSalt(img, percentage):
    img_PS = img
    # get the PepperandSalt number
    NoiseNum = int(img_PS.shape[0] * img_PS.shape[1] * percentage)
    for i in range(NoiseNum):
        # get the random pixel to put PepperandSalt noise
        noise_X = random.randint(0, img_PS.shape[0] - 1)
        noise_Y = random.randint(0, img_PS.shape[1] - 1)
        # put PepperandSalt noise
        if random.random() <= 0.5:
            img_PS[noise_X, noise_Y] = 0
        else:
            img_PS[noise_X, noise_Y] = 255
    return img_PS


if __name__ == '__main__':
    img = cv2.imread('lenna.png', 0)
    img_PepperandSalt = PepperandSalt(img, 0.8)
    plt.subplot(1,2,1)
    plt.title('PepperandSalt Image')
    plt.imshow(img_PepperandSalt, cmap = 'gray')
    plt.axis('off')

    img_org = cv2.imread('lenna.png', 0)
    plt.subplot(1,2,2)
    plt.title('Orginal Image')
    plt.imshow(img_org, cmap = 'gray')
    plt.axis('off')

    plt.show()
