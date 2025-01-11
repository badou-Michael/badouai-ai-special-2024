import cv2 as cv
import numpy as np

img = cv.imread('photo.jpg')

img2 = img.copy()

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

m = cv.getPerspectiveTransform(src, dst)
reslut = cv.warpPerspective(img2, m, (337,488))

cv.imshow("src", img)
cv.imshow("result",result)
cv.waitKey(0)
