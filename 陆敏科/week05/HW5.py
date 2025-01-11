from google.colab import drive
drive.mount('/content/drive')

import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt
import math





# 手搓Canny
if __name__ == '__main__':
  img_path = '/content/drive/MyDrive/lenna.png'
  img = plt.imread(img_path)
  print('image', img)

  if img_path[-4:] == '.png':
    img = img*255

  # 1、灰度化  
  img = img.mean(axis=-1) # 在channel维度进行取均值

  # 2、高斯平滑
  sigma = 0.5
  dim = 5
  Gaussian_filter = np.zeros((dim, dim))
  idx = [i - dim//2 for i in range(dim)] # -2, -1, 0, 1, 2
  n1 = 1/(2*math.pi*sigma**2)
  n2 = -1/(2*sigma**2)
  for i in range(dim):
    for j in range(dim):
      Gaussian_filter[i,j] = n1*math.exp(n2*(idx[i]**2+idx[j]**2))

  Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
  dx, dy = img.shape
  img_new = np.zeros(img.shape)
  tmp = dim//2
  img_pad = np.pad(img, ((tmp, tmp),(tmp, tmp)), 'constant')
  for i in range(dx):
    for j in range(dy):
      img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)
  plt.figure(1)
  plt.imshow(img_new.astype(np.uint8), cmap='gray')
  plt.axis('off')

  # 3、sobel算子，检测水平垂直边缘，并存储梯度
  sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
  sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
  img_tidu_x = np.zeros(img_new.shape)
  img_tidu_y = np.zeros(img_new.shape)
  img_tidu = np.zeros(img_new.shape)
  img_pad = np.pad(img,((1,1),(1,1)),'constant') # 卷积核的大小是1，所以padding=1
  for i in range(dx):
    for j in range(dy):
      img_tidu_x[i,j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_x)
      img_tidu_y[i,j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_y)
      img_tidu[i,j] = np.sqrt(img_tidu_x[i,j]**2 + img_tidu_y[i,j]**2)
  
  img_tidu_x[img_tidu_x == 0] = 0.0000000001
  angle = img_tidu_y / img_tidu_x
  plt.figure(2)
  plt.imshow(img_tidu.astype(np.uint8),cmap='gray')
  plt.axis('off')

  # 4、非极大值抑制
  img_yizhi = np.zeros(img_tidu.shape)
  for i in range(1, dx-1):  # 这里没有padding，所以最外圈是考虑不到的
    for j in range(1, dy-1):
      flag = True
      temp = img_tidu[i-1:i+2, j-1:j+2]
      if angle[i,j] <= -1:
        num_1 = (temp[0,1] - temp[0,0])/angle[i,j] + temp[0,1]
        num_2 = (temp[2,1] - temp[2,2])/angle[i,j] + temp[2,1]
        if not (img_tidu[i,j] > num_1) and (img_tidu[i,j] > num_2):
          flag = False
      elif angle[i,j] >= 1:
        num_1 = (temp[0,2] - temp[0,1])/angle[i,j] + temp[0,1]
        num_2 = (temp[2,0] - temp[2,1])/angle[i,j] + temp[2,1]
        if not (img_tidu[i,j] > num_1) and (img_tidu[i,j] > num_2):
          flag = False
      elif angle[i,j] > 0:
        num_1 = (temp[0,2] - temp[1,2])*angle[i,j] + temp[1,2]
        num_2 = (temp[2,0] - temp[1,0])*angle[i,j] + temp[1,0]
        if not (img_tidu[i,j] > num_1) and (img_tidu[i,j] > num_2):
          flag = False
      elif angle[i,j] < 0:
        num_1 = (temp[1,2] - temp[2,2])*angle[i,j] + temp[1,2]
        num_2 = (temp[1,0] - temp[0,0])*angle[i,j] + temp[1,0]
        if not (img_tidu[i,j] > num_1) and (img_tidu[i,j] > num_2):
          flag = False
      if flag:
        img_yizhi[i,j] = img_tidu[i,j]
  plt.figure(3)
  plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
  plt.axis('off')

  # 5、双阈值检测，边缘连接
  print('tidu_max',np.max(img_yizhi))
  print('tidu_min',np.min(img_yizhi))
  # low_boundary = img_tidu.mean() * 0.6
  low_boundary = 300
  # high_boundary = low_boundary * 3
  high_boundary = 500
  zhan = []
  for i in range(1, img_yizhi.shape[0]-1): #没有padding，外圈同样也不考虑了
    for j in range(1, img_yizhi.shape[1]-1):
      if img_yizhi[i,j] >= high_boundary:
        img_yizhi[i,j] = 255
        zhan.append([i,j])
      elif img_yizhi[i,j] <= low_boundary:
        img_yizhi[i,j] = 0
  
  while not len(zhan) == 0:
    temp_1,temp_2 = zhan.pop()
    a = img_yizhi[temp_1-1:temp_1+2, temp_2-1:temp_2+2]
    if (a[0,0] < high_boundary) and (a[0,0] > low_boundary):
      img_yizhi[temp_1-1, temp_2-1] = 255
      zhan.append([temp_1-1, temp_2-1])
    if (a[0,1] < high_boundary) and (a[0,1] > low_boundary):
      img_yizhi[temp_1-1, temp_2] = 255
      zhan.append([temp_1-1, temp_2])
    if (a[0,2] < high_boundary) and (a[0,2] > low_boundary):
      img_yizhi[temp_1-1, temp_2+1] = 255
      zhan.append([temp_1-1, temp_2+1])
    if (a[1,0] < high_boundary) and (a[1,0] > low_boundary):
      img_yizhi[temp_1, temp_2-1] = 255
      zhan.append([temp_1, temp_2-1])
    if (a[1,2] < high_boundary) and (a[1,2] > low_boundary):
      img_yizhi[temp_1, temp_2+1] = 255
      zhan.append([temp_1, temp_2+1])
    if (a[2,0] < high_boundary) and (a[2,0] > low_boundary):
      img_yizhi[temp_1+1, temp_2-1] = 255
      zhan.append([temp_1+1, temp_2-1])
    if (a[2,1] < high_boundary) and (a[2,1] > low_boundary):
      img_yizhi[temp_1+1, temp_2] = 255
      zhan.append([temp_1+1, temp_2])
    if (a[2,2] < high_boundary) and (a[2,2] > low_boundary):
      img_yizhi[temp_1+1, temp_2+1] = 255
      zhan.append([temp_1+1, temp_2+1])
  
  for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[0]):
      if img_yizhi[i,j] != 0 and img_yizhi[i,j] != 255:
        img_yizhi[i,j] = 0
  
  plt.figure(4)
  plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
  plt.axis('off')
  plt.show()









# 调接口
img = cv2.imread('/content/drive/MyDrive/lenna.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(cv2.Canny(gray,300,500))
cv2.waitKey()
cv2.destroyAllWindows()









# 窗口任意调节Canny下阈值
def CannyThreshold(lowThreshold):
  gray = cv2.GaussianBlur(gray,(3,3),0)
  detected_edges = cv2.Canny(
              gray,
              lowThreshold,
              lowThreshold*ratio,
              apertureSize = kernel_size
              )
  dst = cv2.bitwise_and(img, img, mask=detected_edges)
  cv2_imshow(dst)

lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

img = cv2.imread('/content/drive/MyDrive/lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('canny result')
cv2.createTrackbar('Min threshold','canny result',lowThreshold, max_lowThreshold, CannyThreshold)

CannyThreshold(0)
if cv2.waitKey(0) == 27:
  cv2.destroyAllWindows()










# sobel，laplace，canny接口比较
img = cv2.imread('/content/drive/MyDrive/lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # cv2.CV_64F -- float64
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
img_sobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 1, ksize=3)

img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3) 

img_canny = cv2.Canny(img_gray, 100, 150)

plt.subplot(231), plt.imshow(img_gray, "gray"), plt.title("Original")  
plt.subplot(232), plt.imshow(img_sobel_x, "gray"), plt.title("Sobel_x")  
plt.subplot(233), plt.imshow(img_sobel_y, "gray"), plt.title("Sobel_y")
plt.subplot(234), plt.imshow(img_sobel, "gray"), plt.title("Sobel")
plt.subplot(235), plt.imshow(img_laplace,  "gray"), plt.title("Laplace")  
plt.subplot(236), plt.imshow(img_canny, "gray"), plt.title("Canny")  
plt.show() 
