import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


# 读取照片
img=plt.imread('lenna.png')
plt.subplot(221)
plt.imshow(img)
print(img)

# 灰度化
img_gray=rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')
print(img_gray)

# 二值化
img_binary=np.where(img_gray>=0.5,1,0)
plt.subplot(223)
plt.imshow(img_binary,cmap='gray')
print(img_binary)
plt.show()
