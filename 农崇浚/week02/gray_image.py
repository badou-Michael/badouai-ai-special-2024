from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt
import numpy as np

#原图
img = cv2.imread("girls-583917_640.jpg")
plt.subplot(221)
plt.imshow(img)
plt.title("original image")
plt.axis('off')

#灰度化
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')
plt.title("gray image")
plt.axis('off')

#二值化
img_binary = np.where(img_gray > 0.5, 1, 0)
plt.subplot(223)
plt.imshow(img_binary)
plt.title("binary image")
plt.axis('off')
plt.show()

#print(img)
