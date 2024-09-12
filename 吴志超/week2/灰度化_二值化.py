import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

#image = cv2.imread("lenna.png")
# cv2.imshow("lenna",image)

plt.subplot(221)
img = plt.imread("lenna.png")
# img = cv2.imread("lenna.png", False)
plt.imshow(img)

#灰度化
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray,cmap="gray")

#二值化
img_binary = np.where(img_gray >= 0.5,1,0)
plt.subplot(224)
plt.imshow(img_binary,cmap="gray")
plt.show()
