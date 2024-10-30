import cv2
import numpy as np


img = cv2.imread("../week02/lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("canny", cv2.Canny(gray, 10, 200))  # 第一个参数是需要处理的原图像，该图像必须为单通道的灰度图； 第二个参数是阈值1； 第三个参数是阈值2。
cv2.waitKey()
cv2.destroyAllWindows()





