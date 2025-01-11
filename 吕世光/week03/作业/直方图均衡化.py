import cv2

img = cv2.imread("lenna.png")
(b, g, r) = cv2.split(img)
hist_b = cv2.equalizeHist(b)
hist_g = cv2.equalizeHist(g)
hist_r = cv2.equalizeHist(r)
hist_img = cv2.merge((hist_b, hist_g, hist_r))
cv2.imshow("test hist", hist_img)
cv2.waitKey(0)
