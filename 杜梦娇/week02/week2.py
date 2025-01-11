from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np 
import cv2

#导入图片数据、将图片转为RGB通道、获取图像大小信息并初始化变量
image = cv2.imread("lenna.png")
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h,w,tun_num = image.shape
print(image.shape, image.dtype)
print(img_rgb.shape, img_rgb.dtype)
img_gray = np.zeros((h, w), image.dtype)
img_gray_threshold = np.zeros((h, w), image.dtype)
img_gray1 = np.zeros((h, w), image.dtype)


#灰度化--方法一（直接调用函数进行灰度化）
img_gray = rgb2gray(img_rgb)
print(img_gray.shape, img_gray.dtype)
#灰度化--方法二（阈值灰度化）
img_gray_by_threshold = (0.11*img_rgb[:, :, 0]+0.59*img_rgb[:, :, 1]+0.3*img_rgb[:, :, 2]).astype(image.dtype)
print(img_gray_by_threshold.shape, img_gray_by_threshold.dtype)
#灰度化--方法三（调用cv库进行灰度化）
img_gray1 = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
print(img_gray1.shape, img_gray1.dtype)

#绘制图像
plt.subplot(221)
plt.imshow(img_rgb)
plt.title('RGB tunnel image')
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
plt.title('Gray image(rgb2gray)')
plt.subplot(223)
plt.imshow(img_gray_by_threshold, cmap='gray')
plt.title('Gray image(threshold)')
plt.subplot(224)
plt.imshow(img_gray1, cmap='gray')
plt.title('Gray image(cv)')
plt.show()

#二值化--方法一（阈值法）
thresholded_image = np.where(img_gray > 0.5, 1, 0)
#二值化--方法二（cv阈值法）
ret, cv_thresholded_image = cv2.threshold(img_gray, 0.5, 1.0, cv2.THRESH_BINARY)
#二值化--方法二（cv自适应）
adaptive_thresholded_image = cv2.adaptiveThreshold(img_gray_by_threshold, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 55, 2)
plt.subplot(221)
plt.imshow(thresholded_image, cmap='gray')
plt.title('Threshold')
plt.subplot(222)
plt.imshow(cv_thresholded_image, cmap='gray')
plt.title('CV_threshold')
plt.subplot(223)
plt.imshow(adaptive_thresholded_image, cmap='gray')
plt.title('CV_adaptive_threshold')
plt.show()
