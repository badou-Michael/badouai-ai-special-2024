import cv2
import numpy as np

img = cv2.imread("photo1.jpg")
result = img.copy()

src = np.float32([[207,151],[517,285], [17,601], [343,731]])
dsc = np.float32([[0,0],[337,0],[0,488],[337,488]])

print(img.shape)
warpMatrix = cv2.getPerspectiveTransform(src, dsc)
print("warpMatrix:")
print(warpMatrix)
result1 = cv2.warpPerspective(result, warpMatrix, (337,488))
cv2.imshow("src", img)
cv2.imshow("result", result1)
cv2.waitKey(0)
