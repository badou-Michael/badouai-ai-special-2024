#!/usr/bin/env python
# encoding=gbk
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
必要参数：
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
第二个参数是阈值1；
第三个参数是阈值2。
'''

img = cv2.imread("../week02/lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("canny", cv2.Canny(gray, 200, 300))

'''
手动实现
1.图形灰度化
2.高斯平滑
3.边缘检测 
4.非极大值抑制
5.双阈值算法检测和连接边缘
'''
######  1 .图形灰度化  #################
img = cv2.imread("../week02/lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''
计算高斯核函数
g(x,y)=1/2*PI*sigma^2 *(e^-(x^2+y^2)/(2*sigma^2))
'''
sigma = 0.5
p1 = 1 / (2 * np.pi * sigma ** 2)
p2 = -1 / 2 * sigma ** 2
dim = 5  # 高斯核尺寸
gauss_filter = np.zeros([dim, dim])  # 存储高斯核
# 将X,Y 生成在 [-dim/2,dim/2] i 映射为 tmp [i]中的值
tmp = np.zeros(dim)
for i in range(dim):
    tmp[i] = i - dim // 2

for i in range(dim):
    for j in range(dim):
        gauss_filter[i, j] = p1 * np.exp((tmp[i] ** 2 + tmp[j] ** 2) * p2)

# 归一化高斯核
# _range = np.max(gauss_filter) - np.min(gauss_filter)
# gauss_filter = (gauss_filter - np.min(gauss_filter)) / _range
gauss_filter = gauss_filter / gauss_filter.sum()

# 与原图做卷积
dx, dy = gray.shape
# 存储平滑之后的图像，zeros函数得到的是浮点型数据
img_new = np.zeros(gray.shape)
tmp = dim // 2
# 边缘填补 常数填充
'''
第一个元素(before_1, after_1)表示第一维【列】的填充方式：前面填充before_1个数值，后面填充after_1个数值
第2个元素(before_2, after_2)表示第二维【行】的填充方式：前面填充before_2个数值，后面填充after_2个数值
a = np.array([1, 2, 3, 4, 5])
a=np.pad(a,(2,4),'constant')
[0 0 1 2 3 4 5 0 0 0 0]
      0 0 0 0 0 0 0
      0 0 0 0 0 0 0
      0 0 x x x 0 0
      0 0 x x x 0 0
      0 0 x x x 0 0
      0 0 0 0 0 0 0
      0 0 0 0 0 0 0  
'''
img_gauss_pad = np.pad(gray, ((tmp, tmp), (tmp, tmp)), 'constant')
for i in range(dx):
    for j in range(dy):
        img_new[i, j] = np.sum(img_gauss_pad[i:i + dim, j:j + dim]*gauss_filter)

plt.figure(1)
plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
plt.axis('off')

#######  2、求梯度。sobel矩阵  #############################
'''
梯度f(x,y) =  df(x,y)/dx,df(x,y)/dy
'''
soble_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
soble_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
# 存储梯度图像
img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像
img_tidu_y = np.zeros([dx, dy])
# 梯度值
img_tidu = np.zeros(img_new.shape)
'''
       0 0 0 0 0 
       0 x x x 0 
       0 x x x 0 
       0 x x x 0 
       0 0 0 0 0 
'''
img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
for i in range(dx):
    for j in range(dy):
        img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3]*soble_x)
        img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3]*soble_y)
        img_tidu[i,j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
# 防止 x =0 为无穷大
img_tidu_x[img_tidu_x == 0] = 0.00000001
# 梯度角度
angle = img_tidu_y / img_tidu_x

plt.figure(2)
plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
plt.axis('off')

#######  3、非极大值抑制  #############################
img_yizhi = np.zeros(img_tidu.shape)
for i in range(1, dx - 1):
    for j in range(1, dy - 1):
        flag = True  # 在8邻域内是否要抹去做个标记
        # y=  y = y1*(1-k) +y2*k => y= (y2-y1)k + y1
        temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
        if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
            num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] >= 1:
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

plt.figure(3)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')

# 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
lower_boundary =img_tidu.mean() * 0.5  #128
print(lower_boundary)
high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
zhan = []
for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
    for j in range(1, img_yizhi.shape[1] - 1):
        #标记为强边缘
        if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
            img_yizhi[i, j] = 255
            zhan.append([i, j])
        #像素被抑制
        elif img_yizhi[i, j] <= lower_boundary:  # 舍
            img_yizhi[i, j] = 0

#弱边缘像素  通过查看弱边缘像素及其8个邻域像素，只要其中一个为强边缘像素，则该弱边缘点就可以保留为真实的边缘

while not len(zhan) == 0 :
    temp_1, temp_2 = zhan.pop()  # 出栈
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

#再次抑制
for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[1]):
        if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
            img_yizhi[i, j] = 0

#绘图
plt.figure(4)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')  # 关闭坐标刻度值


plt.show()
