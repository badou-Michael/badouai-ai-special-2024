'''
椒盐噪声
'''
import random
import cv2
import numpy as np

choice = [0, 255]

def gray_func(img, snr, choice):
    new_img = np.copy(img)
    height, width = img.shape[:2]
    pepper_num = int(height * width * snr)
    for i in range(pepper_num):
        randomX = random.randint(0, width-1)
        randomY = random.randint(0, height-1)
        new_img[randomX, randomY] = random.choice(choice)
    return new_img

def bgr_func(img, snr, choice):
    new_img = np.copy(img)
    height, width, channel = img.shape[:3]
    pepper_num = int(height * width * channel * snr)
    for i in range(pepper_num):
        randomC = random.randint(0, channel - 1)
        randomX = random.randint(0, width-1)
        randomY = random.randint(0, height-1)
        new_img[randomX, randomY, randomC] = random.choice(choice)
    return new_img

snr = 0.01
pth = "lenna.png"
img = cv2.imread(pth)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
new_gray = gray_func(gray_img, snr, choice)
new_bgr = bgr_func(img, snr, choice)
cv2.imshow("Image", img)
cv2.imshow("Gray", new_gray)
cv2.imshow("Bgr", new_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
