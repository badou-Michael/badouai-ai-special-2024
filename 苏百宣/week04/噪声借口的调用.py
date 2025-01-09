# 噪声借口的调用
# author：苏百宣

import cv2 as cv
import numpy as np
from skimage import util

# 读取原始图像
img = cv.imread("lenna.png")

# 添加泊松噪声，返回的值在 [0, 1] 之间
noise_gs_img = util.random_noise(img, mode='poisson')

# 将噪声图像转换为 [0, 255] 之间的 uint8 格式
noise_gs_img = np.array(255 * noise_gs_img, dtype='uint8')

# 显示原图
cv.imshow("source", img)

# 显示带有噪声的图像
cv.imshow("lenna", noise_gs_img)

# 等待按键并关闭窗口
cv.waitKey(0)
cv.destroyAllWindows()
