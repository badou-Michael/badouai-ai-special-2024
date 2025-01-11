import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
canny——边缘检测函数
cv2.Canny(image, threshold1, threshold2, edges=None, apertureSize=3, L2gradient=False)
threshold1, threshold2——>双阈值（高阈值和低阈值）
apertureSize 是 Sobel 算子（用于计算图像梯度）的大小。它必须是 1、3、5 或 7。默认值是 3，这意味着 Sobel 卷积核的大小是 3x3。
'''
img = cv2.imread('F:\DeepLearning\Code_test\lenna.png',1)

gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
canny_img = cv2.Canny(gray_img,100,200,apertureSize=3)

# 创建一个空的彩色图像用于叠加边缘
color_edges = np.zeros_like(img)
# 将边缘检测结果（二值图像）叠加到彩色图像的对应通道上
# 将边缘叠加到所有三个通道上
color_edges[canny_img != 0] = img[canny_img != 0]

plt.subplot(231),plt.imshow(img),plt.title('src')
plt.subplot(232),plt.imshow(canny_img,'gray'),plt.title('canny_gray')
plt.subplot(233),plt.imshow(color_edges),plt.title('canny_color')
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()
