from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

''' histogram equalization '''

#灰度图均衡化
imgGrey = cv2.imread("lenna.png", 0)
hist = cv2.calcHist([imgGrey],[0],None,[256],[0,256])
imgGrey2 = cv2.equalizeHist(imgGrey)
hist2 = cv2.calcHist([imgGrey2],[0],None,[256],[0,256])

cv2.imshow("Histogram Equalization", np.hstack([imgGrey, imgGrey2]))
cv2.waitKey(0)

plt.figure()
plt.title("equalizational Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.xlim([0,256])
plt.plot(hist,color="red")
plt.plot(hist2,color="green")
plt.show()

#彩色图均衡化
img = cv2.imread("lenna.png", 1)
(b,g,r) = cv2.split(img)
imgB = cv2.equalizeHist(b)
imgG = cv2.equalizeHist(g)
imgR = cv2.equalizeHist(r)
img2 = cv2.merge((imgB,imgG,imgR))

cv2.imshow("Histogram Equalization 3", np.hstack([img, img2]))
cv2.waitKey(0)
