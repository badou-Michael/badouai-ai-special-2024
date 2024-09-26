import numpy as np
import cv2
def GaussianNoise(src, means, sigma, percentage):
    NoiseImg = src.copy()  # 使用 copy 避免直接修改原始图像  
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    rows, cols = src.shape
    indices = np.random.randint(0, rows * cols, size=NoiseNum)
    randX = indices // cols
    randY = indices % cols

    # 使用 np.random.normal 生成噪声  
    noise = np.random.normal(means, sigma, size=NoiseNum)

    # 确保噪声后的像素值在有效范围内  
    NoiseImg[randX, randY] = np.clip(NoiseImg[randX, randY] + noise, 0, 255).astype(np.uint8)

    return NoiseImg
  
# 读取图像
img = cv2.imread('lenna.png', 0)  # 灰度模式读取  
img_noisy = GaussianNoise(img, 2, 4, 0.8)

# 显示图像  
cv2.imshow('Original Image', img)
cv2.imshow('Gaussian Noise Image', img_noisy)
cv2.waitKey(0)
cv2.destroyAllWindows()
