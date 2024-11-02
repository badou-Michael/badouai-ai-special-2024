import cv2
import random


def gaussian_noise(src_image, sigma, mean, percentage):
    noise_image = src_image
    h, w = noise_image.shape[:2]
    num = int(h * w * percentage)
    for i in range(num):
        # 每次提取一个随机的点，因为将图片看成是平面，所以取x和y定位到这个点，然后在这个点上的像素值基础上再加上高斯随机数
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)
        noise_image[x, y] += random.gauss(mean, sigma)
        if noise_image[x, y] < 0:
            noise_image[x, y] = 0
        elif noise_image[x, y] > 255:
            noise_image[x, y] = 255
    return noise_image


if __name__ == '__main__':
    image = cv2.imread("../week02/lenna.png", 0)
    image1 = gaussian_noise(image, 2, 4, 0.8)
    image = cv2.imread("../week02/lenna.png")
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('source', image2)
    #在文件夹中写入图片
    cv2.imwrite('lenna_GaussianNoise.png',image1)
    cv2.imshow('lenna_GaussianNoise', image1)
    cv2.waitKey(0)
