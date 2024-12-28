import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, std=25):
    """
    添加高斯噪声
    :param image: 输入图像
    :param mean: 噪声的均值
    :param std: 噪声的标准差
    :return: 添加噪声后的图像
    """
    # 生成高斯噪声
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    
    # 将噪声添加到图像中
    noisy_image = cv2.add(image, noise)
    
    return noisy_image

# 示例使用
image = cv2.imread('input.jpg')
noisy_image = add_gaussian_noise(image, mean=0, std=25)
cv2.imwrite('output_gaussian_noise.jpg', noisy_image)

def add_salt_and_pepper_noise(image, prob=0.05):
    """
    添加椒盐噪声
    :param image: 输入图像
    :param prob: 噪声的概率
    :return: 添加噪声后的图像
    """
    noisy_image = np.copy(image)
    h, w = image.shape[:2]
    
    # 生成随机噪声
    noise = np.random.rand(h, w)
    
    # 添加椒噪声（黑色）
    noisy_image[noise < prob / 2] = 0
    
    # 添加盐噪声（白色）
    noisy_image[noise > 1 - prob / 2] = 255
    
    return noisy_image

# 示例使用
image = cv2.imread('input.jpg')
noisy_image = add_salt_and_pepper_noise(image, prob=0.05)
cv2.imwrite('output_salt_and_pepper_noise.jpg', noisy_image)
