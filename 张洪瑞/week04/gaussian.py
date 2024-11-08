'''
高斯噪声
'''
import random
import cv2
import numpy as np

# 灰度图像高斯噪声
def func_gray(img, sigma, mean, percent):
    gauss = random.gauss(mu=mean, sigma=sigma)
    height, width = img.shape[:2]
    pixnum = int(percent * height * width);
    new_img = np.copy(img)
    for i in range(pixnum):
        randomX = random.randint(0, width-1)
        randomY = random.randint(0, height-1)
        new_img[randomX, randomY] = new_img[randomX, randomY] + gauss
        if new_img[randomX, randomY] < 0:
            new_img[randomX, randomY] = 0
        elif new_img[randomX, randomY] > 255:
            new_img[randomX, randomY] = 255
    return new_img

# 彩色图像高斯噪声
def func_rgb(img, sigma, mean, percent):
    gauss = random.gauss(mu=mean, sigma=sigma)
    height, width, channel = img.shape[:3]
    new_img = np.copy(img)
    pixnum = int(percent * height * width * channel);
    for i in range(pixnum):
        randomC = random.randint(0, channel-1)
        randomX = random.randint(0, width-1)
        randomY = random.randint(0, height-1)
        new_img[randomX, randomY, randomC] = new_img[randomX, randomY, randomC] + gauss
        if new_img[randomX, randomY, randomC] < 0:
            new_img[randomX, randomY, randomC] = 0
        elif new_img[randomX, randomY, randomC] > 255:
            new_img[randomX, randomY, randomC] = 255
    return new_img

means = 10
sigma = 0.5
pth = "lenna.png"
img = cv2.imread(pth)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gauss_gary_img = func_gray(img_gray, means, sigma, 0.9)
gauss_bgr_img = func_rgb(img, means, sigma, 0.9)
cv2.imshow("Gray_Img", gauss_gary_img)
cv2.imshow("Bgr_Img", gauss_bgr_img)
cv2.imshow("Img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
