import cv2
import numpy as np
from skimage import util
img = cv2.imread("lenna.png")
noise_img=util.random_noise(img,mode='s&p',amount = 0.3)
noise_img = (noise_img * 255).astype(np.uint8)
cv2.imwrite('lenna_noise.png',noise_img)

def aHash(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    hash = ''
    avg = np.average(img)
    for i in range (8):
        for j in range (8):
            if img[i,j] > avg:
                hash += '1'
            else:
                hash += '0'
    return hash

def dHash(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.resize(img, (9,8), interpolation=cv2.INTER_AREA)
    hash = ''
    for i in range(8):
        for j in range(8):
            if img[i, j] > img[i + 1, j]:
                hash += '1'
            else:
                hash += '0'
    return hash

def compHash(hash1, hash2):
    n = 0
    if len(hash1) == 0 or len(hash1) != len(hash2):
        return -1
    else:
        for i in range(len(hash1)):
            if hash1[i] != hash2[i]:
                n += 1
    
    return n

aHash1 = aHash(img)
aHash2 = aHash(noise_img)
dHash1 = dHash(img)
dHash2 = dHash(noise_img)
a_hash_res = compHash(aHash1, aHash2)
d_hash_res = compHash(dHash1, dHash2)
print(a_hash_res, d_hash_res)
