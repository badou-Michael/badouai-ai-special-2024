import cv2
import numpy as np

def differnce_hash(img_path):
    image =cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    hash =0
    for i in range(8):
        for j in range(8):
            if(image[i,j]>127):
                hash |=1 <<(63 -i *8-j)

    return hash

def average_hash(img_path):
    image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    avg = np.mean(image)
    hash=0
    for i in range(8):
        for j in range(8):
            if image[i,j]>avg:
                hash |=1<<(63-i*8-j)
    return  hash


img1 = 'lenna.png'
img2 = 'lenna.png'

hash1 = differnce_hash(img1)
hash2 = differnce_hash(img2)

print(hash1)
print(hash2)

hash3 = average_hash(img1)
hash4 = average_hash(img2)

print(hash3)
print(hash4)
