'''
yzc-canny
'''
import cv2

img = cv2.imread("..\\lenna.png",1)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("yzc_canny:",cv2.Canny(img_gray,10,100))
cv2.waitKey()
cv2.destroyAllWindows()
