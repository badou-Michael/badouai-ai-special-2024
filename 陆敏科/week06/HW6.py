from google.colab import drive
drive.mount('/content/drive')

from google.colab.patches import cv2_imshow
import numpy as np
import cv2


# 透视变换矩阵求解
def WarpPerspectiveMatrix(src, dst):
  assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

  nums = src.shape[0]
  A = np.zeros((2*nums, 8))
  B = np.zeros((2*nums, 1))
  for i in range(0, nums):
    A_i = src[i,:]
    B_i = dst[i,:]
    A[2*i,:] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
    A[2*i+1,:] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
    B[2*i] = B_i[0]
    B[2*i+1] = B_i[1]

  A = np.mat(A)
  warpMatrix = A.I * B   # 矩阵运算
  warpMatrix = np.array(warpMatrix).T[0]
  warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)   # 加入a33=1
  warpMatrix = warpMatrix.reshape((3, 3))
  return warpMatrix


if __name__ == '__main__':
    print('warpMatrix:')
    print()
    src = np.array([[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]])
    dst = np.array([[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]])
    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)










# 透视变换接口调用
img = cv2.imread('/content/drive/MyDrive/photo1.jpg')
result = img.copy()
src = np.float32([[207,151],[517,285],[17,601],[343,731]])
dst = np.float32([[0,0],[337,0],[0,488],[337,488]])

print(img.shape)

m = cv2.getPerspectiveTransform(src, dst)
print('WarpMatrix:',m)

result = cv2.warpPerspective(result,m,(337,488))
cv2_imshow(img)
cv2_imshow(result)










# KMeans接口
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = np.array([[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]
    ])
cls = KMeans(n_clusters=3)
y_pred = cls.fit_predict(X)
print(cls)
print(y_pred)

colors = ['red','blue','green']
labels = ['A', 'B', 'C']

for i in range(3):
  plt.scatter(X[y_pred == i,0], X[y_pred == i,1], c=colors[i], label=labels[i], marker='x')

plt.title('KMeans_Basketball_Data')
plt.xlabel('assists_per_minute')
plt.ylabel('points_per_minute')
plt.legend()
plt.show()












# cv2.kmeans接口处理灰度图
import matplotlib.pyplot as plt

img = cv2.imread('/content/drive/MyDrive/lenna.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

h,w = gray_img.shape
data = gray_img.reshape((h*w, 1))
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, flags)
dst = labels.reshape((gray_img.shape[0], gray_img.shape[1]))
plt.rcParams['font.sans-serif']=['SimHei']

titles = [u'原始图像', u'聚类图像']
images = [gray_img, dst]
for i in range(2):
  plt.subplot(1,2,i+1), plt.imshow(images[i],'gray')
  plt.title(titles[i])
  plt.xticks([]), plt.yticks([])
plt.show()












# cv2.kmeans接口处理BGR图
img = cv2.imread('/content/drive/MyDrive/lenna.png')
data = img.reshape((-1, 3))
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

#聚集成2类
compactness, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)
#聚集成4类
compactness, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)
#聚集成8类
compactness, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)
#聚集成16类
compactness, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)
#聚集成64类
compactness, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)

centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst2 = res.reshape((img.shape))

centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]
dst4 = res.reshape((img.shape))

centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape((img.shape))

centers16 = np.uint8(centers16)
res = centers16[labels16.flatten()]
dst16 = res.reshape((img.shape))

centers64 = np.uint8(centers64)
res = centers64[labels64.flatten()]
dst64 = res.reshape((img.shape))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

plt.rcParams['font.sans-serif']=['SimHei']

titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4',
          u'聚类图像 K=8', u'聚类图像 K=16',  u'聚类图像 K=64']  
images = [img, dst2, dst4, dst8, dst16, dst64]  
for i in range(6):  
   plt.subplot(2,3,i+1), plt.imshow(images[i], 'gray'), 
   plt.title(titles[i])  
   plt.xticks([]),plt.yticks([])  
plt.show()
