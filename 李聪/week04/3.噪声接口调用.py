import cv2 as cv
import numpy as np
from skimage import util

# 读取图像
img = cv.imread("lenna.png")

# 1. 添加高斯噪声
noise_gaussian_img = util.random_noise(img, mode='gaussian')
noise_gaussian_img = np.array(255 * noise_gaussian_img, dtype=np.uint8)  # 将浮点型数据转换为整型

# 2. 添加椒盐噪声 
noise_pepperand_salt_img = util.random_noise(img, mode='s&p')
noise_pepperand_salt_img = np.array(255 * noise_pepperand_salt_img, dtype=np.uint8)  # 将浮点型数据转换为整型

# 显示原始图像
cv.imshow("Original Image", img)

# 显示高斯噪声图像
cv.imshow("Gaussian Noise", noise_gaussian_img)

# 显示椒盐噪声图像 (PepperandSalt)
cv.imshow("PepperandSalt Noise", noise_pepperand_salt_img)

cv.waitKey(0)
cv.destroyAllWindows()
