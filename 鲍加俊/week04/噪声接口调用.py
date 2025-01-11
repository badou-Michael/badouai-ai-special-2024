import cv2
import numpy as np

def add_gaussian_noise_opencv(image, mean=0, std=25):
    """
    使用OpenCV添加高斯噪声
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
noisy_image = add_gaussian_noise_opencv(image, mean=0, std=25)
cv2.imwrite('output_gaussian_noise_opencv.jpg', noisy_image)
