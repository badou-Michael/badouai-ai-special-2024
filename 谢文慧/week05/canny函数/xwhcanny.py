import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
  # img = cv2.imread("lenna.png")
  # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  # cv2.imshow("canny",cv2.Canny(gray,100,200))
  # cv2.waitKey()
  img = cv2.imread("lenna.png")
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  # print(gray)
  # 1、高斯平滑
  sigma = 0.5
  dim = 5
  # 存储高斯核，这是数组不是列表了
  Gaussian_filter = np.zeros([dim,dim])
  # 生成一个序列,-2,-1,0,1,2,表示中心相邻几个点的距离位置
  tmp = [i-dim//2 for i in range(dim)]
  # 计算高斯核,用二维高斯函数
  n1 = 1/(2*math.pi*sigma**2)
  n2 = -1/(2*sigma**2)
  # 计算高斯滤波中每一个点的值
  for i in range(dim):
      for j in range(dim):
          Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
  # 归一化后的高斯滤波,让所有值加起来得1
  Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
  # 图像形状，高度 宽度
  print(gray.shape)
  dx,dy = gray.shape
  # 存储高斯滤波平滑之后的图像
  img_new = np.zeros(gray.shape)
  # 计算边界填补的像素数量
  tmp = dim//2
  img_pad = np.pad(gray, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘填补
  # 对每一个点，用高斯滤波计算后得到的值
  for i in range(dx):
      for j in range(dy):
          img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)
  # print(img_new)
  # print("---------------------")
  # print(img_new.astype(np.uint8))
  plt.subplot(231)
  plt.title("Gaussian_filter  hot figure")
  plt.imshow(img_new.astype(np.uint8))
  plt.subplot(232)
  plt.title("Gaussian_filter  gray picture")
  plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
  # 取消坐标
  plt.axis('off')
  # https://blog.csdn.net/a435262767/article/details/107115249/?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-4--blog-19766615.235^v43^pc_blog_bottom_relevance_base5&spm=1001.2101.3001.4242.3&utm_relevant_index=7
  
  # 2、求梯度。以下两个是滤波用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
  # 水平
  sobel_kernel_x  = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
  #垂直
  sobel_kernel_y  = np.array([[1,2,1,],[0,0,0],[-1,-2,-1]])
  # 存储两个滤波计算后的图像
  # ？为什么要叫成梯度？
  img_tidu_x = np.zeros([dx, dy])
  img_tidu_y = np.zeros([dx, dy])
  
  img_tidu = np.zeros(img_new.shape)
  # 边缘填补，根据上面矩阵结构3维，所以是1
  img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')
  for i in range(dx):
      for j in range(dy):
          img_tidu_x[i,j] = np.sum(img_pad[i:i+3,j:j+3]*sobel_kernel_x)
          img_tidu_y[i, j]= np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)
          # 两个方向综合作用结果，平方开根号
          img_tidu[i, j] = np.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)
  img_tidu_x[img_tidu_x == 0] = 0.00000001
  plt.subplot(233)
  plt.title("sobel picture")
  plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
  plt.axis('off')
  # https://blog.csdn.net/weixin_45858794/article/details/106159560
  # https: // blog.csdn.net / greatwall_sdut / article / details / 104571526 /
  
  # 计算tan，为后续
  angle = img_tidu_y/img_tidu_x
  
  # 3、非极大值抑制
  img_yizhi = np.zeros(img_tidu.shape)
  for i in range(1, dx - 1):
      for j in range(1, dy - 1):
          # 在8邻域内是否要抹去做个标记
          flag = True
          # 梯度幅值的8邻域矩阵
          temp = img_tidu[i - 1:i + 2, j - 1:j + 2]
          if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
              # 单线性差值公式
              num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
              num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
              if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                  flag = False
          elif angle[i, j] >= 1:  # 说明夹角大于45度，
              num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
              num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
              if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                  flag = False
          elif angle[i, j] > 0:
              num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
              num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
              if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                  flag = False
          elif angle[i, j] < 0:
              num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
              num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
              if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                  flag = False
          if flag:
              img_yizhi[i, j] = img_tidu[i, j]
  # print(img_tidu)
  # print("---------")
  # print(img_yizhi)
  plt.subplot(234)
  plt.title("not max value delete picture")
  plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
  plt.axis('off')
  
  # 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
  # 设定最低阈值是均值的0.5倍
  lower_boundary = img_tidu.mean() * 0.5
  # 设定高阈值是低阈值的3倍
  high_boundary = lower_boundary * 3
  zhan = []
  for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
      for j in range(1, img_yizhi.shape[1] - 1):
          if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
              img_yizhi[i, j] = 255
              zhan.append([i, j])
          elif img_yizhi[i, j] <= lower_boundary:  # 舍
              img_yizhi[i, j] = 0
  
  while not len(zhan) == 0:
      temp_1, temp_2 = zhan.pop()  # 出栈
      # print(zhan)
      # print("---------")
      # print(temp_1, temp_2)
      a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
      if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
          img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
          zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
      if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
          img_yizhi[temp_1 - 1, temp_2] = 255
          zhan.append([temp_1 - 1, temp_2])
      if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
          img_yizhi[temp_1 - 1, temp_2 + 1] = 255
          zhan.append([temp_1 - 1, temp_2 + 1])
      if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
          img_yizhi[temp_1, temp_2 - 1] = 255
          zhan.append([temp_1, temp_2 - 1])
      if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
          img_yizhi[temp_1, temp_2 + 1] = 255
          zhan.append([temp_1, temp_2 + 1])
      if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
          img_yizhi[temp_1 + 1, temp_2 - 1] = 255
          zhan.append([temp_1 + 1, temp_2 - 1])
      if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
          img_yizhi[temp_1 + 1, temp_2] = 255
          zhan.append([temp_1 + 1, temp_2])
      if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
          img_yizhi[temp_1 + 1, temp_2 + 1] = 255
          zhan.append([temp_1 + 1, temp_2 + 1])
  
  for i in range(img_yizhi.shape[0]):
      for j in range(img_yizhi.shape[1]):
          if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
              img_yizhi[i, j] = 0
  
      # 绘图]
  plt.subplot(235)
  plt.title("canny picture")
  plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
  plt.axis('off')  # 关闭坐标刻度值
  plt.show()
