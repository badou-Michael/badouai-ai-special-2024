# Canny是目前最优秀的边缘检测算法之一，其目标为找到一个最优的边缘，其最优边缘的定义为：
# 1、好的检测：算法能够尽可能的标出图像中的实际边缘
# 2、好的定位：标识出的边缘要与实际图像中的边缘尽可能接近
# 3、最小响应：图像中的边缘只能标记一次

# Canny 实现步骤

# 1. 对图像进行[灰度化]
# 2. 对图像进行[高斯滤波]：
# 根据待滤波的像素点及其邻域点的灰度值按照一定的参数规则进行加权平均。这样可以有效滤去理想图像中叠加的高频噪声。
# 3. 检测图像中的水平、垂直和对角边缘（如[Prewitt，Sobel算子]等）。
# 4. 对梯度幅值进行[非极大值抑制]：通俗意义上是指寻找像素点局部最大值，将非极大值点所对应的灰度值置为0，这样可以剔除掉一大部分非边缘的点。
# 5. 用[双阈值算法]检测和连接边缘

import math
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pic_path = 'lenna.png'
    img = plt.imread(pic_path)
    print("image",img)
    
    if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255 # 还是浮点数类型
    img = img.mean(axis=-1)  # 取均值的方法进行灰度化


    # 1、高斯平滑
    sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
    dim = 5  # 高斯核尺寸
    
    Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核，这是数组不是列表了
    tmp = [i-dim//2 for i in range(dim)]  # 生成一个序列
    n1 = 1/(2*math.pi*sigma**2)  # 计算高斯核
    n2 = -1/(2*sigma**2)
    
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    dx, dy = img.shape
    img_new = np.zeros(img.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    tmp = dim//2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘填补
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)
    
    # 显示高斯平滑后的图像
    plt.figure('1.Gaussian')
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')
    
    # 2、求梯度。以下两个是滤波用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_x)
            img_tidu_y[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y/img_tidu_x
    plt.figure('2.梯度')
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')
    
    
    # 3、非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx-1):
        for j in range(1, dy-1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_tidu[i-1:i+2, j-1:j+2]  # 梯度幅值的8邻域矩阵
            angle_ij = angle[i, j]
            if angle_ij <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / angle_ij + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle_ij + temp[2, 1]
            elif angle_ij >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle_ij + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle_ij + temp[2, 1]
            elif angle_ij > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle_ij + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle_ij + temp[1, 0]
            elif angle_ij < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle_ij + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle_ij + temp[1, 2]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]

    plt.figure('3.非极大值抑制')
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []

    # 使用布尔索引来加速阈值检测
    strong_edges = (img_yizhi >= high_boundary)
    weak_edges = (img_yizhi <= lower_boundary)
    img_yizhi[strong_edges] = 255
    img_yizhi[weak_edges] = 0

    # 将强边缘点的坐标加入栈中
    zhan.extend(np.argwhere(strong_edges))

    while zhan:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_yizhi[temp_1-1:temp_1+2, temp_2-1:temp_2+2]
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue
                ni, nj = temp_1 + di, temp_2 + dj
                if lower_boundary < img_yizhi[ni, nj] < high_boundary:
                    img_yizhi[ni, nj] = 255
                    zhan.append([ni, nj])

    # 将非边缘点设置为0
    img_yizhi[(img_yizhi != 0) & (img_yizhi != 255)] = 0

    # 绘图
    plt.figure('4.双阈值检测')
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()
