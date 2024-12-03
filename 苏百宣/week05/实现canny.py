# 实现canny
# author：苏百宣

import cv2

img = cv2.imread("sww1028.jpg", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("sww1028", cv2.Canny(gray, 300, 20))
cv2.waitKey()
cv2.destroyAllWindows()
