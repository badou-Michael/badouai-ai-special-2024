import cv2

oriimg = cv2.imread("lenna.png")
grayimg = cv2.cvtColor(oriimg,cv2.COLOR_BGR2GRAY)
cny_img = cv2.Canny(grayimg,100,200,apertureSize=3)
cv2.imshow("cny_img",cny_img)
cv2.waitKey(0)
cv2.destroyAllWindows()