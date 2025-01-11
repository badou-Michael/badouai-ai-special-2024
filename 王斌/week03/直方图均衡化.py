
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("C:/Users/Administrator/Desktop/123.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dst = cv2.equalizeHist(gray)
hist = cv2.calcHist([dst],[0],None,[256],[0,255])

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", dst)
cv2.waitKey(0)
