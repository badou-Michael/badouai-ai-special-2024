import cv2
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

image = cv2.imread('lenna.png')
if image is None:
    print("Error")
else:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #灰度化
    binary_img = np.where(gray_image >= 128, 255, 0)      #二值化，其中0.5,1,0报错，经过老师指导改为128,255,0

    # 打印图像 #

    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(132)
    plt.imshow(gray_image, cmap="gray")
    plt.title('Gray Image')

    plt.subplot(133)
    plt.imshow(binary_img, cmap="gray")
    plt.title('Binary Image')

    plt.tight_layout()
    plt.show()
