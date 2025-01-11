import random
import cv2
import numpy as np


def gaussian_noise(image, means, sigma, percentage):
    image_dict = {}
    noise_image = image
    noise_number = int(percentage * image.shape[0] * image.shape[1])
    for i in range(noise_number):
        rand_x = random.randint(0, image.shape[0] - 1)
        rand_y = random.randint(0, image.shape[0] - 1)
        if rand_x in image_dict:
            while len(image_dict[rand_x]) == image.shape[1]:
                rand_x = random.randint(0, image.shape[0] - 1)
            while rand_y in image_dict[rand_x]:
                rand_y = random.randint(0, image.shape[0] - 1)
            image_dict[rand_x].append(rand_y)
        else:
            image_dict[rand_x] = [rand_y]
        noise_image[rand_x, rand_y] = noise_image[rand_x, rand_y] + random.gauss(means, sigma)
        if noise_image[rand_x, rand_y] < 0:
            noise_image[rand_x, rand_y] = 0
        elif noise_image[rand_x, rand_y] > 255:
            noise_image[rand_x, rand_y] = 255
    return noise_image


img = cv2.imread('lenna.png', 0)
img1 = cv2.imread('lenna.png',0)
img2 = gaussian_noise(img1, 2, 8, 0.8)
cv2.imshow('img + lenna_GaussianNoise', np.hstack([img, img2]))
cv2.waitKey(0)
