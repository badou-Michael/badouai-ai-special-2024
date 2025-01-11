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


# 还不太行，cp拖累了np的速度，但是cp的速度也不快，可能是因为数据量太小了

import math
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

def gaussian_filter(dim, sigma):
    tmp = np.arange(-dim//2 + 1., dim//2 + 1.)
    x, y = np.meshgrid(tmp, tmp)
    Gaussian_filter = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    Gaussian_filter /= Gaussian_filter.sum()
    return Gaussian_filter

def apply_filter(img, filter):
    img_pad = np.pad(img, ((filter.shape[0]//2, filter.shape[0]//2), (filter.shape[1]//2, filter.shape[1]//2)), 'constant')
    img_new = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_new[i, j] = np.sum(img_pad[i:i+filter.shape[0], j:j+filter.shape[1]] * filter)
    return img_new

if __name__ == '__main__':
    pic_path = 'lenna.png'
    img = plt.imread(pic_path)
    print("image", img)
    
    if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型
    img = img.mean(axis=-1)  # 取均值的方法进行灰度化

    # 1、高斯平滑
    sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
    dim = 5  # 高斯核尺寸
    
    # 生成高斯核
    Gaussian_filter = gaussian_filter(dim, sigma)
    
    # 使用卷积进行高斯平滑
    img_new = apply_filter(img, Gaussian_filter)
    
    # 显示高斯平滑后的图像
    plt.figure('1.Gaussian')
    plt.imshow(img_new.astype(np.uint8), cmap='gray')
    plt.axis('off')
    
    # 将图像数据从NumPy数组转换为CuPy数组
    img_new_cp = cp.array(img_new)

    # 2、求梯度。以下两个是滤波用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    sobel_kernel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = cp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = cp.zeros_like(img_new_cp)  # 存储梯度图像
    img_tidu_y = cp.zeros_like(img_new_cp)
    img_tidu = cp.zeros_like(img_new_cp)
    img_pad = cp.pad(img_new_cp, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
    for i in range(img_new_cp.shape[0]):
        for j in range(img_new_cp.shape[1]):
            img_tidu_x[i, j] = cp.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_x)
            img_tidu_y[i, j] = cp.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_y)
            img_tidu[i, j] = cp.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y / img_tidu_x
    plt.figure('2.梯度')
    plt.imshow(cp.asnumpy(img_tidu).astype(np.uint8), cmap='gray')
    plt.axis('off')
    
    # 3、非极大值抑制
    img_yizhi = cp.zeros_like(img_tidu)
    for i in range(1, img_new_cp.shape[0] - 1):
        for j in range(1, img_new_cp.shape[1] - 1):
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
    plt.imshow(cp.asnumpy(img_yizhi).astype(np.uint8), cmap='gray')
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
    zhan.extend(cp.argwhere(strong_edges))

    while zhan:
        temp_1, temp_2 = zhan.pop()  # 出栈
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
    plt.imshow(cp.asnumpy(img_yizhi).astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()
'''
1.高斯平滑：

使用NumPy生成高斯核和进行卷积操作，因为这些操作在CPU上已经足够高效。
将卷积后的图像转换为CuPy数组，以便后续在GPU上进行计算。

2.梯度计算：

使用CuPy进行Sobel滤波和梯度计算，这些操作在GPU上可以显著加速。

3.非极大值抑制：

使用CuPy进行非极大值抑制，因为这是一个计算密集型操作，适合在GPU上并行化。

4.双阈值检测：

使用CuPy进行双阈值检测和边缘连接，这些操作在GPU上可以显著加速。
'''

'''
适合使用CuPy的情况：

1.大规模数据处理：
当处理的数据量非常大时，GPU的并行计算能力可以显著提高计算速度。
例如，大型矩阵运算、卷积操作、傅里叶变换等。

2.重复性高的计算：
当需要进行大量重复性高的计算时，GPU的并行计算能力可以显著提高效率。
例如，深度学习中的前向传播和反向传播、图像处理中的滤波操作等。

3.矩阵和向量运算：
CuPy在矩阵和向量运算方面具有显著优势，特别是涉及大量矩阵乘法、加法等操作时。
---------------------------------------------------------------------

适合使用NumPy的情况：

1.小规模数据处理：
当处理的数据量较小时，CPU的计算能力已经足够，使用NumPy可以避免GPU初始化和数据传输的开销。
例如，小型数组运算、简单的统计计算等。

2.非并行化操作：
某些操作不适合并行化，或者并行化带来的性能提升不明显，使用NumPy更为合适。
例如，涉及大量条件判断、复杂逻辑控制的操作。

3.数据传输频繁：
当需要频繁在CPU和GPU之间传输数据时，数据传输的开销可能会抵消并行计算带来的性能提升，使用NumPy更为合适。
-----------------------------------------------------------------------------------------------

结合使用CuPy和NumPy：
在实际应用中，可以结合使用CuPy和NumPy，充分利用两者的优势。例如：

使用CuPy进行大规模矩阵运算和卷积操作。
使用NumPy进行小规模数据处理和复杂逻辑控制。
在必要时，将数据从CuPy数组转换为NumPy数组，或从NumPy数组转换为CuPy数组。
'''

''' [example]
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

# 示例数据
data = np.random.rand(1000, 1000)

# 将数据从NumPy数组转换为CuPy数组
data_cp = cp.array(data)

# 使用CuPy进行大规模矩阵运算
result_cp = cp.dot(data_cp, data_cp.T)

# 将结果从CuPy数组转换回NumPy数组
result_np = cp.asnumpy(result_cp)

# 使用NumPy进行小规模数据处理
mean_value = np.mean(result_np)

print("Mean value:", mean_value)

'''