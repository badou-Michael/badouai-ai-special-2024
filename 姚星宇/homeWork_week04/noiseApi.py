import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util

# 读取图像
image_path = 'lenna.png'  # 确保lenna.png在当前目录下
image = io.imread(image_path) / 255  # 将图像像素值归一化到 [0, 1] 范围

# 1. 添加高斯噪声
gaussian_noisy_image = util.random_noise(image, mode='gaussian', mean=0, var=0.5)

# 2. 添加椒盐噪声
salt_pepper_noisy_image = util.random_noise(image, mode='s&p', amount =0.5)

# 显示图像
plt.figure(figsize=(12, 4))

# 原始图像
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 高斯噪声图像
plt.subplot(1, 3, 2)
plt.imshow(gaussian_noisy_image, cmap='gray')
plt.title('Gaussian Noisy Image')
plt.axis('off')

# 椒盐噪声图像
plt.subplot(1, 3, 3)
plt.imshow(salt_pepper_noisy_image, cmap='gray')
plt.title('Salt & Pepper Noisy Image')
plt.axis('off')

# 显示所有子图
plt.show()