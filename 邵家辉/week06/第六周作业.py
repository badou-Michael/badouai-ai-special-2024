import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1.透视变换
img = cv2.imread('photo1.jpg')
img_copy = img.copy()
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [512, 0], [0, 512], [512, 512]])
m = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(img_copy, m, (512, 512))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)

# 2.K-means算法
X = [[1,3],[4,2],[7,6],[99,33],[4,2],[7,7],[8,8],[12,66],[43,67],[32,54],[23,5],[22,3],[32,5],[66,3],[3,6],[66,99]]
clf = KMeans(n_clusters=3, n_init='auto')
y_predict = clf.fit_predict(X)
print(y_predict)
x1 = []
y1 = []
x2 = []
y2 = []
x3 = []
y3 = []

for index,value in enumerate(X):
    if y_predict[index] == 0:
        x1.append(value[0])
        y1.append(value[1])
    elif y_predict[index] == 1:
        x2.append(value[0])
        y2.append(value[1])
    else:
        x3.append(value[0])
        y3.append(value[1])

plt.scatter(x1, y1, color='red', label='A')
plt.scatter(x2, y2, color='blue', label='B')
plt.scatter(x3, y3, color='green', label='C')
plt.xlabel("hello")
plt.ylabel("ggg")
plt.legend()
plt.show()
