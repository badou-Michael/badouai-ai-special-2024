from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

# 原图
plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)

# 灰度化
plt.subplot(223)
img_gray=rgb2gray(img)
plt.imshow(img_gray,cmap='gray')

# 二值化
plt.subplot(224)
img_binary=np.where(img_gray>=0.5,1,0)
plt.imshow(img_binary,cmap="gray")
plt.show()
