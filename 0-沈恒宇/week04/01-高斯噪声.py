# 随机生成符合正态（高斯）分布的随机数，means，sigma为两个函数
import cv2
import numpy as np

# 生成高斯噪声
"""
1.创建方法
2.取图像的随机坐标，
3.给图像的随机坐标加上随机整数
4.像素值边界判断
5.使用方法
"""


def gaussian_noise(image, means, sigma, prob):
    noise_image = image.copy()
    # 获取需要加噪声的像素的数量
    noise_number = int(prob*image.size)
    # 随机获取需要加噪声的像素索引
    noise_indices = np.random.choice(image.size, noise_number, replace=False)
    # 将索引转化为坐标
    noise_coords = np.unravel_index(noise_indices, image.shape)
    # 给图像的像素值加上高斯随机数
    # noise_image[noise_coords] += np.random.normal(means, sigma, noise_number).astype(np.uint8)
    noise_image[noise_coords] += np.random.normal(means, sigma, noise_number).astype(np.uint8)
    # 使用numpy的clip函数进行边界判断
    noise_image = np.clip(image, 0, 255)
    return noise_image


if __name__ == '__main__':
    image_gray = cv2.imread('lenna.png', 0)
    image_noise = gaussian_noise(image_gray, 2, 4, 0.8)
    cv2.imshow('image_gary',image_gray)
    cv2.imshow('image_noise',image_noise)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
