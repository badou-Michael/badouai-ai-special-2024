import cv2
import random

def salt_papper_noise(img, snr):
    np = int(img.shape[0]*img.shape[1]*snr)

    for i in range(np):
        x_r = random.randint(0,img.shape[0]-1)
        y_r = random.randint(0,img.shape[1]-1)

        if random.random() < 0.5:
            img[x_r, y_r] = 0
        else:
            img[x_r, y_r] = 255
    return img


def GaussNoise(src, means, sigma):
    NoiseImg = src
    #NoiseNum = int(pencetage*src.shape[0]*src.shape[1])

    for x in range(int(src.shape[0])):
        for y in range(int(src.shape[1])):
            NoiseImg[x, y] = NoiseImg[x, y] + random.gauss(means,sigma)
            if NoiseImg[x, y] < 0:
                NoiseImg[x, y] = 0
            elif NoiseImg[x, y] > 255:
                NoiseImg[x, y] = 255
    return NoiseImg
