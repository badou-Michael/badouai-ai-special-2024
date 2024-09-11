import cv2
import random

def func(src, percentage):
    NoiseImg = img
    NoiseNum = int(percentage*img.shape[0]*img.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, img.shape[0]-1)
        randY = random.randint(0, img.shape[1]-1)
        if random.random() >= 0.5:
            NoiseImg[randX, randY] = 255
        else:
            NoiseImg[randX, randY] = 0
    return NoiseImg

img = cv2.imread('lenna.png', 0)
img1 = func(img, 0.2)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('noise', img1)
cv2.imshow('sorce', img2)
cv2.waitKey()
