import cv2
import numpy as np

def resize_image(image,width=8,height=8):
    resized_image = cv2.resize(image,(width,height))
    return resized_image

def compute_hash_mean(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    hash_image = 1 * (gray > mean)
    return hash_image


def compute_hash_diff(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    resized_gray = resize_image(gray,9,8)
    diff = resized_gray[:,1:] > resized_gray[:,:-1]
    return diff

def hamming_distance(hash1,hash2):
    return np.count_nonzero(hash1 != hash2)


image1 = cv2.imread('lenna.png')
image1.shape

mean = 0
sigma = 50
height, width, channels = image1.shape
gaussian_noise = np.random.normal(mean, sigma, (height, width, channels)).astype(np.uint8)
image2 = cv2.add(image1, gaussian_noise)

resized_image1 = resize_image(image1)
resized_image2 = resize_image(image2)

hash1 = compute_hash_mean(resized_image1)
hash2 = compute_hash_mean(resized_image2)

distance = hamming_distance(hash1,hash2)
distance
#14


hash1 = compute_hash_diff(image1)
hash2 = compute_hash_diff(image2)
distance = hamming_distance(hash1.astype(int),hash2.astype(int))
distance
#20
