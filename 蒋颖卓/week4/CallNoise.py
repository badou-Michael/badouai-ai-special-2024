import numpy as np
import cv2

import numpy as np
import cv2


def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)

    # 随机生成噪声
    total_pixels = image.size
    num_salt = np.ceil(salt_prob * total_pixels).astype(int)
    num_pepper = np.ceil(pepper_prob * total_pixels).astype(int)

    # 添加盐噪声
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 1  # 1表示白色

    # 添加椒噪声
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0  # 0表示黑色

    return noisy_image

def add_gaussian_noise(image, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), gaussian_noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)  # 确保值在[0, 255]范围内
    return noisy_image


if __name__ == '__main__':
    # 读取图像
    image = cv2.imread('C:/Users/DMR/Desktop/1.png', cv2.IMREAD_GRAYSCALE)

    # 添加椒盐噪声
    salt_and_pepper_noise_image = add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02)

    # 添加高斯噪声
    gaussian_noise_image = add_gaussian_noise(image, mean=0, sigma=25)

    # 显示结果
    cv2.imshow('Original Image', image)
    cv2.imshow('Salt and Pepper Noise', salt_and_pepper_noise_image)
    cv2.imshow('Gaussian Noise', gaussian_noise_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
