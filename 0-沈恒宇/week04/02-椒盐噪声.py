"""
添加椒盐噪声

1.方法
2.获取数量
3.随机获取坐标
4.添加噪声
"""
import cv2
import numpy as np


def sally_pepper_noise(image, prob):
    image_noise = image.copy()
    # 计算噪声的数量
    noise_number = int(image.size*prob)
    # 获随机索引
    noise_index = np.random.choice(image.size, noise_number, replace=False)
    # 将索引转换为坐标
    image_coords = np.unravel_index(noise_index, image.shape)
    # 添加噪声
    image_noise[image_coords] = np.random.randint(0, 2)*255
    return image_noise


if __name__ == '__main__':
    image_gray = cv2.imread('lenna.png',0)
    noise_image = sally_pepper_noise(image_gray, 0.8)
    cv2.imshow('original image', image_gray)
    cv2.imshow('noise_image', noise_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
