# 彩色图像灰度化&二值化
# 20240905 徐凯

'''
逻辑思考：
1、导入cv、plt库
2、cv库导入图片
3、cv库转化图片
4、plt输出图片
'''

import cv2
import matplotlib.pyplot as plt

# 原图
img = cv2.imread('lenna.png')
plt.subplot(221)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

# 灰度化
plt.subplot(222)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray, cmap='gray')

# 二值化
plt.subplot(223)
_, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
plt.imshow(img_binary, cmap='gray')

plt.show()
