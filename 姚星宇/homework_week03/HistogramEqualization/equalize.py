from matplotlib import pyplot as plt
import cv2
import numpy as np

img = cv2.imread("./lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

equal = cv2.equalizeHist(gray)
plt.figure()
plt.hist(equal.ravel(), 256)
plt.show()
cv2.imshow("lenna", equal)
cv2.waitKey(0)
cv2.destroyAllWindows()