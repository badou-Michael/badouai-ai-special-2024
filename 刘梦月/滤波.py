import cv2
import numpy as np
import matplotlib.pyplot as plt

# 生成一个带有椒盐噪声的示例图像
image = np.random.rand(100, 100).astype(np.float32)
image[::10, ::10] = 0 # 人为引入椒盐噪声，每隔10个像素有一个黑点

# 使用 OpenCV 中的 medianBlur 函数进行中值滤波
filtered_image = cv2.medianBlur((image * 255).astype(np.uint8), 3)

# 显示原始和滤波后的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image with Noise')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Filtered Image (Median Filter)')
plt.imshow(filtered_image, cmap='gray')
plt.show()