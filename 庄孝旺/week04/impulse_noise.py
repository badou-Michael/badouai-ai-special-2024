import random
import cv2
import numpy as np


def impulse_noise(image, percentage):
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
        if random.random() <= 0.5:
            noise_image[rand_x, rand_y] = 0
        else:
            noise_image[rand_x, rand_y] = 255
    return noise_image


img = cv2.imread('lenna.png', 0)
img1 = cv2.imread('lenna.png',0)
img2 = impulse_noise(img1, 0.2)
cv2.imshow('img + lenna_ImpulseNoise', np.hstack([img, img2]))
cv2.waitKey(0)
