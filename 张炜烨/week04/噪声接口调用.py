import cv2 as cv
from skimage import util

# 读取图像
img = cv.imread('lenna.png', 0)

# 添加椒盐噪声
noisy_img = util.random_noise(img, mode='s&p', amount=0.05)

# 将噪声图像转换为 0-255 范围的 uint8 类型
noisy_img = (255 * noisy_img).astype('uint8')

# 保存结果
cv.imwrite('lenna_salt_and_pepper2.png', noisy_img)