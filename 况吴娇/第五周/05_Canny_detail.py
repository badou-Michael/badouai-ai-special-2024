'''
常量
math.pi：圆周率π的值。
math.e：自然对数的底数e的值。
函数
math.sqrt(x)：计算x的平方根。
math.pow(x, y)：计算x的y次幂。
math.sin(x)：计算x的正弦值（x是以弧度为单位）。
math.cos(x)：计算x的余弦值（x是以弧度为单位）。
math.tan(x)：计算x的正切值（x是以弧度为单位）。
math.asin(x)：计算x的反正弦值，并返回弧度。
math.acos(x)：计算x的反余弦值，并返回弧度。
math.atan(x)：计算x的反正切值，并返回弧度。
math.log(x)：计算x的自然对数。
math.log10(x)：计算x的以10为底的对数。
math.floor(x)：返回小于或等于x的最大整数。
math.ceil(x)：返回大于或等于x的最小整数。
math.fabs(x)：返回x的绝对值。
math.exp(x)函数用来计算e的x次幂
'''
import numpy as np
import matplotlib.pyplot as plt
import math
#Canny边缘检测算法包括几个关键步骤：高斯平滑、梯度计算、非极大值抑制和双阈值检测。
if __name__ == '__main__':
    pic_path = 'lenna.png'
    img = plt.imread(pic_path)
    print("image", img)
    if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型
    img = img.mean(axis=-1)  # 取均值的方法进行灰度化
#mean(axis=-1) 计算最后一个轴（即通道轴）上的平均值，有效地将每个像素的三个值合并为一个灰度值。

# 1、高斯平滑
##公式：G(x,y)=1/2πσ^2 * e *(−(x^2+y^2)/2σ^2 ) #G(x,y) 是高斯核函数，σ 是标准差，控制平滑的程度。x 和 y 是相对于核中心的位置

# sigma = 1.52  # 高斯平滑时的高斯核参数，标准差，可调
    sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
    dim = 5  # 高斯核尺寸
    Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核，这是数组不是列表了
##dim//2 是 dim 除以 2 的向下取整结果，这意味着如果 dim 是偶数，结果是 dim/2；如果 dim 是奇数，结果是 (dim-1)/2。
#在Canny边缘检测算法中，这个列表用于生成高斯核的中心线。高斯核是一个对称的矩阵，其中心线是对称轴。
    tmp = [i - dim // 2 for i in range(dim)]  # 生成一个序列
#这个列表 tmp 通常用于表示高斯核的中心线偏移量。在高斯核中，dim//2 是核的中心（因为 dim 通常是奇数，所以 dim//2 就是中心像素的索引）。
# 列表 tmp 中的每个元素都表示核中的每个元素相对于中心的偏移量。
    n1 = 1 / (2 * math.pi * sigma ** 2)  # 计算高斯核
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()#Gaussian_filter / Gaussian_filter.sum()：将高斯滤波器中的每个元素除以其总和，从而实现归一化。
    #Gaussian_filter / Gaussian_filter.sum()：将高斯滤波器中的每个元素除以其总和，从而实现归一化。
    dx, dy = img.shape
    img_new = np.zeros(img.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    tmp = dim // 2##dim//2 就是半径，即核中心到边缘的距离
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘填补 'constant': 用一个常数值填充（默认为0）。
    # 第一个 (tmp, tmp) 指定了第一个轴（垂直轴，即图像的行）的填充：第一个 tmp 表示在轴的开始处（图像的顶部）填充 tmp 个像素，
    #                  第二个 tmp 表示在轴的结束处（图像的底部）填充 tmp 个像素。
    # 第二个 (tmp, tmp) 指定了第二个轴（水平轴，即图像的列）的填充：同样地，第一个 tmp 表示在轴的开始处（图像的最左边）填充 tmp 个像素
    #                  ，第二个 tmp 表示在轴的结束处（图像的最右边）填充 tmp 个像素。
    #填充的目的是为了在对图像边缘进行卷积时，不会出现索引越界的错误。
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
#当需要将处理后的图像显示或保存时，需要将其转换为uint8格式，因为大多数图像查看器和处理软件都期望像素值是整数，并且范围在0到255之间。
#    plt.axis('off') ##plt.axis('off') 关闭坐标轴，使图像显示更为清洁。

    # 2、求梯度。以下两个是滤波用的sobel矩阵（检测图像中的水平、垂直和对角边缘）## 通过梯度，我们可以检测边缘、提取特征、增强图像、估计运动和深度，以及进行图像配准等。
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像
    img_tidu_y = np.zeros([dx, dy])# #np.zeros(img_new.shape) 灰度图像，一样的
    img_tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补
    # 使用 np.pad 对图像进行边缘扩展，这里扩展了1个像素。这是因为Sobel算子核的大小是3x3，如果不进行边缘扩展，卷积操作会在图像边缘处越界。
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # x方向
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # y方向
            img_tidu[i, j] = np.sqrt(
                img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)  # np.sqrt 计算每个像素点的梯度幅度，这是通过计算x方向和y方向梯度值的平方和的平方根得到的。
    img_tidu_x[img_tidu_x == 0] = 0.00000001  # ，它将所有被选取出来的等于零的梯度值（即x方向上没有边缘的点）赋值为一个非常小的正数（0.00000001）
    angle = img_tidu_y / img_tidu_x  # 计算每个像素点的梯度方向，这是通过梯度的y分量除以x分量得到的。
    # 显示梯度图像
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 3、非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            # dx 和 dy 分别是图像的宽度和高度。range(1, dx-1) 和 range(1, dy-1) 确保循环从第二行/列开始，到倒数第二行/列结束，从而跳过了图像的最外层边缘。
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
            # 这行代码提取了当前像素点 (i, j) 周围的8个邻域像素的梯度值。temp 是一个3x3的矩阵，包含了这些梯度值。
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
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []
    for i in range(1, img_yizhi.shape[0]-1):  # 外圈不考虑了
        for j in range(1, img_yizhi.shape[1]-1):
            if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
                img_yizhi[i, j] = 0
    while not len(zhan) == 0:  # while not len(zhan) == 0: 这个循环会一直执行，直到 zhan 栈变空。
        temp_1, temp_2 = zhan.pop()  # 出栈 这行代码 temp_1, temp_2 = zhan.pop() 是因为栈 zhan 中存储的是像素点的坐标，每个坐标由一对 (x, y) 值组成
        # 。# pop() 操作总是返回最后添加到栈中的元素。
        # temp_1 就存储了 x 坐标，temp_2 存储了 y 坐标。
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]  ##3x3的邻域矩阵 a
        ## a= img_yizhi[temp_1-1:temp_1+2, temp_2-1:temp_2+2] 这行提取了当前处理的强边缘点周围的3x3邻域
        # 然后，代码检查这个邻域中的每个像素点，如果它们的梯度值在低阈值和高阈值之间，
        # 就将它们标记为边缘（设置为255），并将它们的坐标加入到栈 zhan 中，以便后续处理。
        '''a[0,0]  a[0, 1] a[0, 2]
           a[1, 0] a[1, 1] a[1, 2]
           a[2, 0] a[2, 1] a[2, 2]
        '''  # 在这个图示中，a[1,1] 代表中心像素点 (temp_1, temp_2)，其他元素代表其周围的邻居像素点。

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
            zhan.append([temp_1 + 1,
                         temp_2 + 1])  # 这些索引允许我们访问中心像素点周围的邻居，并根据它们的梯度值来决定是否将它们标记为边缘点。这是非极大值抑制和边缘连接过程中的关键步骤，确保了边缘的连续性和完整性。

    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0
        # 代码中的双重循环遍历整个 img_yizhi 图像，将那些既不是0（非边缘）也不是255（强边缘）的像素点设置为0。这是为了确保只有被确认为边缘的点才会被保留在最终的边缘检测结果中。
        # 绘图]
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()
















