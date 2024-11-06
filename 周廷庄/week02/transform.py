import cv2
from skimage.color import rgb2gray

# 使用 cv2 读取图片
img = cv2.imread('lenna.png')

# 将 BGR 图像转换为 RGB 格式（因为 skimage 的 rgb2gray 期望 RGB 输入）
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 灰度化
gray_img = rgb2gray(img_rgb)

# 保存灰度图像
cv2.imwrite('lenna_gray.png', (gray_img * 255).astype('uint8'))

# 二值化（使用阈值 0.5）
binary_img = gray_img > 0.5

# 保存二值图像
cv2.imwrite('lenna_binary.png', (binary_img.astype('uint8') * 255))