import cv2
import numpy as np
import random
from skimage import util


def gaussian_noise(img, mean, sigma, percentage):
    # 获得需要加噪的像素点个数
    noise_number = int(percentage * img.shape[0] * img.shape[1])
    # NumPy数组是可变对象，img_noise = img的写法将在后续修改img_noise中的元素时修改img，将会导致三幅图都出现噪声
    img_noise = img.copy()
    for i in range(noise_number):
        # 随机选择范围内的像素点加噪
        random_x = random.randint(0, img.shape[0] - 1)
        random_y = random.randint(0, img.shape[1] - 1)
        img_noise[random_x, random_y] = img[random_x, random_y] + random.gauss(mean, sigma)
        # 加噪完的像素值可能超出范围，需要处理
        if img_noise[random_x, random_y] > 255:
            img_noise[random_x, random_y] = 255
        elif img_noise[random_x, random_y] < 0:
            img_noise[random_x, random_y] = 0
    return img_noise


def salt_pepper(img, percentage):
    # 获得需要加噪的像素点个数
    noise_number = int(percentage * img.shape[0] * img.shape[1])
    img_noise = img.copy()
    for i in range(noise_number):
        # 随机选择范围内的像素点加噪
        random_x = random.randint(0, img.shape[0] - 1)
        random_y = random.randint(0, img.shape[1] - 1)
        # 使得像素点随即成为椒噪声和盐噪声
        random_mark = random.random()
        if random_mark <= 0.5:
            img_noise[random_x, random_y] = 0
        else:
            img_noise[random_x, random_y] = 255
    return img_noise


if __name__ == "__main__":
    lenna_gray = cv2.imread("lenna.png", 0)
    # 自定义函数加噪
    lenna_noise1 = gaussian_noise(lenna_gray, 2, 4, 0.8)
    lenna_noise2 = salt_pepper(lenna_gray, 0.8)
    cv2.imshow("Lenna in gray", lenna_gray)
    cv2.imshow("Lenna with gaussian noise", lenna_noise1)
    cv2.imshow("Lenna with salt pepper noise", lenna_noise2)
    cv2.waitKey(0)
    # 通过skimage库中的util模块的random_noise接口实现加噪
    lenna_noise3 = util.random_noise(lenna_gray, mode="gaussian", mean=2, var=4)
    lenna_noise4 = util.random_noise(lenna_gray, mode="s&p", amount=0.8)
    cv2.imshow("Lenna in gray", lenna_gray)
    cv2.imshow("Lenna with gaussian noise by skimage", lenna_noise1)
    cv2.imshow("Lenna with salt pepper noise by skimage", lenna_noise2)
    cv2.waitKey(0)




