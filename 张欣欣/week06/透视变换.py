import cv2
import numpy as np

img = cv2.imread("photo1.jpg")
result1 =img.copy()

src = np.float32([[206,154],[517,286],[16,602],[341,733]])
drt = np.float32([[0,0],[300,0],[0,400],[300,400]])
print(img.shape)
m =cv2.getPerspectiveTransform(src,drt)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result1,m,[300,400])
cv2.imshow("src",img)
cv2.imshow("result",result)
cv2.waitKey(0)
