import cv2
img = cv2.imread('E:/GUO_APP/GUO_AI/picture/yang1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("canny", cv2.Canny(gray, 100, 200))
cv2.waitKey()
cv2.destroyAllWindows()
