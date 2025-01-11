import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("C:/Users/bq-twenty-one/Desktop/001.jpg")
plt.figure(1)
plt.imshow(img)
src = np.float32([[39.5, 149.3], [152.4, 151.7], [127.1, 224.3],[227.5, 227.1]])
dst = np.float32([[0, 0], [110, 0], [88, 78], [188, 78]])
img1 = img.copy()
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(img1, m, (188, 78))
plt.figure(2)
plt.imshow(result)
plt.show()
