import cv2
import numpy as np

# 添加高斯噪声
def add_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))  # 生成高斯噪声
    noisy_image = image + gauss  # 给原图像加上噪声
    noisy_image = np.clip(noisy_image, 0, 255)  # 限制像素值在[0, 255]之间
    return noisy_image.astype(np.uint8)

# 读取图像并添加高斯噪声
image = cv2.imread('lenna.png')
noisy_image = add_gaussian_noise(image)

# 显示图像
cv2.imshow('Gaussian Noise', noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
