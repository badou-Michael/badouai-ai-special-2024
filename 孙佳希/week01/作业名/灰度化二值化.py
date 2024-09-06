from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img = plt.imread("lenna.png") #plt读取的图像为RGB图像，且像素值转变为了0到1之间的浮点数
plt.subplot(221)
plt.imshow(img)
print("---image lenna----")
print(img)

# 灰度化
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
