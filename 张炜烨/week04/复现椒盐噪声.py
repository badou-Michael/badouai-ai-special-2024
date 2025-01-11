import numpy as np
import cv2
import random

def salt_and_pepper_noise(image, prob):
    noisy_img = image.copy()
    num_salt = np.ceil(prob * image.size * 0.5)
    num_pepper = np.ceil(prob * image.size * 0.5)

    # Add salt noise
    for _ in range(int(num_salt)):
        i = random.randint(0, image.shape[0] - 1)
        j = random.randint(0, image.shape[1] - 1)
        noisy_img[i, j] = 255

    # Add pepper noise
    for _ in range(int(num_pepper)):
        i = random.randint(0, image.shape[0] - 1)
        j = random.randint(0, image.shape[1] - 1)
        noisy_img[i, j] = 0

    return noisy_img

img = cv2.imread('lenna.png', 0)
noisy_img = salt_and_pepper_noise(img, 0.05)
cv2.imwrite('lenna_salt_and_pepper.png', noisy_img)