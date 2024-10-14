import numpy as np
import cv2


def add_gaussian_noise(image, mean=0, sigma=15):
    """添加高斯噪声到图像"""
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + gaussian_noise, 0, 255)
    return noisy_image.astype(np.uint8)


def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.02):
    """添加椒盐噪声到图像"""
    noisy_image = np.copy(image)
    # 添加盐噪声
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255  # 设置随机位置盐的颜色
    # 添加胡椒噪声
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0  # 设置随机位置胡椒的颜色

    return noisy_image.astype(np.uint8)


# 读取一张示例图像
image = cv2.imread('sea.jpg')  # 替换为图像路径

# 添加高斯噪声
gaussian_noisy_image = add_gaussian_noise(image)
resized_noisy_image1 = cv2.resize(gaussian_noisy_image, (1599, 877))  # 指定新的宽度和高度
cv2.imshow("gaussian_img", resized_noisy_image1)

# 添加椒盐噪声
salt_and_pepper_noisy_image = add_salt_and_pepper_noise(image)
resized_noisy_image2 = cv2.resize(salt_and_pepper_noisy_image, (1599, 877))  # 指定新的宽度和高度
cv2.imshow("sap_img", resized_noisy_image2)
cv2.waitKey(0)