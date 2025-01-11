from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Module1:RGB2GRAY
path = "lenna.png"
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# height, weight, channel
h, w = image.shape[:2]
img_gray = np.zeros([h, w], dtype=image.dtype)
img_gray2 = rgb2gray(image)
# Gray=R0.3+G0.59+B0.11
for row in range(h):
    for col in range(w):
        channel = image[row, col]
        img_gray[row, col] = int(0.3*channel[0] + 0.59*channel[1] + 0.11*channel[2])
cv2.imshow('Image Window', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# image binaryzation
#img_binary = np.where(img_gray >= 0.5, 1, 0)
img_bin = np.zeros([h, w], dtype=np.uint8)
for row in range(h):
    for col in range(w):
        if img_gray[row, col] > 128:
            img_bin[row, col] = 255;
        else:
            img_bin[row, col] = 0;
cv2.imshow('Image Window', img_bin)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.figure()
# 创建2行2列的子图，显示第1张图片
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('OD')
plt.axis('off')
plt.subplot(2, 2, 2)
# 如果不加cmap='gray'则默认三通道显示！
plt.imshow(img_gray, cmap='gray')
plt.title('Gray Image1')
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(img_gray2, cmap='gray')
plt.title('Gray Image2')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(img_bin, cmap='gray')
plt.title('Bin Image')
plt.axis('off')
plt.show()
