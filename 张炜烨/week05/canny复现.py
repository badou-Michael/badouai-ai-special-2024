import numpy as np 
import matplotlib.pyplot as plt
import math 

pic_path = 'lenna.png'
img = plt.imread(pic_path)
print("image",img) 
if pic_path[-4:] == '.png':  
    img *= 255
#img = img.mean(axis=-1)#在最后一个轴上计算均值起到灰度的效果
img = np.dot(img[...,:3], [0.299, 0.587, 0.114])#按权重转换灰度

#第一步：高斯平滑
sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
dim = 5  # 高斯核尺寸
Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核
tmp = [i-dim//2 for i in range(dim)]  #生成高斯核的索引序列 dim = 5，那么生成的序列将是 [-2, -1, 0, 1, 2]
n1 = 1/(2*math.pi*sigma**2)  # 计算二维高斯核  公式为G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-((x^2 + y^2) / (2 * sigma^2)))
n2 = -1/(2*sigma**2)
for i in range(dim):#计算二维高斯滤波器的值
    for j in range(dim):
        Gaussian_filter[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()#归一化
dx, dy = img.shape
img_new = np.zeros(img.shape) 
img_tidu = np.zeros(img_new.shape)
img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant') #为了卷积 做一下边缘填充
for i in range(dx):#卷积操作
    for j in range(dy):
        img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)
plt.figure(1)
plt.imshow(img_new.astype(np.uint8), cmap='gray')  
plt.axis('off')
#求梯度
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像
img_tidu_y = np.zeros([dx, dy])
img_tidu = np.zeros(img_new.shape)
img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
for i in range(dx):
    for j in range(dy):
        img_tidu_x[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_x)  # x方向
        img_tidu_y[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)  # y方向
        img_tidu[i, j] = np.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)
img_tidu_x[img_tidu_x == 0] = 0.00000001
angle = img_tidu_y/img_tidu_x
plt.figure(2)
plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
plt.axis('off')

# 3、非极大值抑制
img_yizhi = np.zeros(img_tidu.shape)
for i in range(1, dx-1):
    for j in range(1, dy-1):
        flag = True  
        temp = img_tidu[i-1:i+2, j-1:j+2]  
        if angle[i, j] <= -1:  
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
# 4、双阈值检测，连接边缘。
lower_boundary = img_tidu.mean() * 0.5
high_boundary = lower_boundary * 3 
zhan = []
#这里遍历图像中的每个像素点，如果像素点的值大于高阈值，则认为它是边缘点，并将其值设置为 255，同时将其坐标添加到栈中。如果像素点的值小于低阈值，则认为它不是边缘点，并将其值设置为 0。
for i in range(1, img_yizhi.shape[0]-1):  
    for j in range(1, img_yizhi.shape[1]-1):
        if img_yizhi[i, j] >= high_boundary: 
            img_yizhi[i, j] = 255
            zhan.append([i, j])
        elif img_yizhi[i, j] <= lower_boundary:  
            img_yizhi[i, j] = 0
#这里使用栈来连接边缘点。每次出栈一个点，然后检查其 8 邻域中的每个点，如果邻域中的点的值在高阈值和低阈值之间，则认为它是边缘点，并将其值设置为 255，同时将其坐标添加到栈中。
while not len(zhan) == 0:
    temp_1, temp_2 = zhan.pop()  
    a = img_yizhi[temp_1-1:temp_1+2, temp_2-1:temp_2+2]
    if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
        img_yizhi[temp_1-1, temp_2-1] = 255  
        zhan.append([temp_1-1, temp_2-1]) 
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

for i in range(img_yizhi.shape[0]):#这里最终处理图像，将所有非边缘点的值设置为 0。
    for j in range(img_yizhi.shape[1]):
        if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
            img_yizhi[i, j] = 0

# 绘图]
plt.figure(4)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')  
plt.show()
