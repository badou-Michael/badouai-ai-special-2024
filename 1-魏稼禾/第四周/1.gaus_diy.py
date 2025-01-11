import cv2
import numpy as np
import random
import copy

def gaussion(img, means, sigma, percentage):
    img_new = copy.deepcopy(img)
    gaus_num = int(img.shape[0]*img.shape[1]*percentage)
    for _ in range(gaus_num):
        ix = random.randint(0, img.shape[0]-1)
        iy = random.randint(0, img.shape[1]-1)
        img_new[ix,iy] = np.clip(int(img_new[ix,iy]+random.gauss(means, sigma)),0,255)
    # img_new = img.astype(np.float32)
    # noise = np.random.normal(means, sigma, img.shape)
    # img_noisy = img_new + noise
    return img_new

def salt_and_pepper(img, percentage):
    img_new = copy.deepcopy(img)
    noise_num = int(percentage*img.shape[0]*img.shape[1])
    for _ in range(noise_num):
        ix = random.randint(0, img.shape[0]-1)
        iy = random.randint(0, img.shape[1]-1)
        if random.random()>0.5:
            img_new[ix,iy]=255
        else:
            img_new[ix,iy]=0
    return img_new

img = cv2.imread("lenna.png",0)
img_gauss = gaussion(img, 2,4,1)
img_SP = salt_and_pepper(img, 0.1)
cv2.imshow("ori", img)
cv2.imshow("gau", img_gauss)
cv2.imshow("SP", img_SP)
cv2.waitKey()
cv2.destroyAllWindows()