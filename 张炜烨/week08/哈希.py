import cv2
import numpy as np

def aHash(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg = np.mean(gray)
    hash_str = ''.join(['1' if pixel > avg else '0' for pixel in gray.flatten()])
    return hash_str

def dHash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''.join(['1' if gray[i, j] > gray[i, j + 1] else '0'
                        for i in range(8) for j in range(8)])
    return hash_str

def cmpHash(hash1, hash2):
    if len(hash1) != len(hash2):
        return -1
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')

hash1 = aHash(img1)
hash2 = aHash(img2)
print(f'aHash: {hash1}, {hash2}')
print('Average Hash similarity:', cmpHash(hash1, hash2))

hash1 = dHash(img1)
hash2 = dHash(img2)
print(f'dHash: {hash1}, {hash2}')
print('Difference Hash similarity:', cmpHash(hash1, hash2))