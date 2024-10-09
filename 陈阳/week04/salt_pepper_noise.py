import cv2
import random


def salt_pepper_noise(src_image, percentage):
    salt_pepper_image = src_image
    h, w = salt_pepper_image.shape[:2]
    number = int(h * w * percentage)
    for i in range(number):
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)
        if random.random() <= 0.5:
            salt_pepper_image[x, y] = 0
        else:
            salt_pepper_image[x, y] = 255
    return salt_pepper_image


if __name__ == "__main__":
    image = cv2.imread("../week02/lenna.png", 0)
    image1 = salt_pepper_noise(image, 0.8)
    cv2.imwrite('lenna_PepperSalt.png', image1)

    image = cv2.imread("../week02/lenna.png")
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('source', image2)
    cv2.imshow('lenna_PepperSalt', image1)
    cv2.waitKey(0)
