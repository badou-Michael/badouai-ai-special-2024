import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("image.jpg")
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

dst = cv2.equalizeHist(image_gray)
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure()
plt.hist(dst.ravel(),256)
plt.show()

cv2.imshow("1",np.hstack([image_gray,dst]))
cv2.waitKey(0)