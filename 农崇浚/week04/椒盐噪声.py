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


img = cv2.imread('lenna.png',0)
cv2.imshow('jiao-yan',img)

img1 = cv2.imread('lenna.png',0)
img2 = salt_papper_noise(img1,0.1)
cv2.imshow('salt_papper',img2)



cv2.waitKey(0)
