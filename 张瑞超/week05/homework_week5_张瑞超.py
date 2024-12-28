import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

# 手写canny
def canny_edge_detection(image_path, sigma=0.5, low_threshold_ratio=0.5, high_threshold_ratio=3.0,
                         padding_mode='constant'):
    """
    实现 Canny 边缘检测算法的函数，使用加权平均法进行灰度化。

    参数:
    - image_path (str): 图像的文件路径。
    - sigma (float): 高斯平滑的标准差，用于控制模糊的程度。默认值为 0.5。
    - low_threshold_ratio (float): 双阈值检测中的低阈值比率。低阈值为图像梯度平均值的乘积。默认值为 0.5。
    - high_threshold_ratio (float): 双阈值检测中的高阈值比率，高阈值为低阈值的倍数。默认值为 3.0。
    - padding_mode (str): 在卷积操作中使用的图像填充模式，例如 'constant', 'edge' 等。默认值为 'constant'。

    返回:
    - numpy.ndarray: 返回处理后的二值化图像，其中边缘的像素值为 255，其他区域为 0。
    """

    # 读取图像
    img = plt.imread(image_path)

    # 如果是 .png 格式的图片，像素值范围为 0-1，需要转换为 0-255
    if image_path[-4:] == '.png':  # 处理 PNG 格式图像
        img = img * 255  # 将图像数据扩展到 255

    # 使用加权平均法进行灰度化处理
    # 权重：R=0.299, G=0.587, B=0.114
    img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    # 1. 高斯平滑处理，减少图像噪声
    dim = int(np.round(6 * sigma + 1))  # 高斯滤波器的维度，根据 sigma 计算
    if dim % 2 == 0:  # 确保高斯核维度为奇数
        dim += 1
    Gaussian_filter = np.zeros([dim, dim])  # 初始化高斯滤波器
    tmp = [i - dim // 2 for i in range(dim)]  # 生成维度的序列，用于高斯核计算
    n1 = 1 / (2 * math.pi * sigma ** 2)  # 高斯核计算公式的前部分
    n2 = -1 / (2 * sigma ** 2)  # 高斯核公式中的指数部分
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))  # 计算高斯核
    Gaussian_filter /= Gaussian_filter.sum()  # 归一化高斯核

    # 对图像进行填充，边缘填充，防止卷积时越界
    dx, dy = img.shape
    img_new = np.zeros(img.shape)  # 创建一个新图像，用于存储平滑后的结果
    tmp = dim // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), padding_mode)  # 对原图像进行填充

    # 进行高斯滤波，平滑图像
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)  # 高斯平滑

    # 2. 梯度计算，使用 Sobel 算子计算图像梯度
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel X 算子
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Sobel Y 算子
    img_tidu_x = np.zeros(img_new.shape)  # 存储 X 方向的梯度
    img_tidu_y = np.zeros(img_new.shape)  # 存储 Y 方向的梯度
    img_tidu = np.zeros(img_new.shape)  # 存储梯度幅值
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), padding_mode)  # 再次填充图像以处理边界

    # 对图像应用 Sobel 算子计算梯度
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # X 方向梯度
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # Y 方向梯度
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)  # 计算梯度幅值

    # 防止除以 0 的情况，将 X 梯度中为 0 的值设为一个极小值
    img_tidu_x[img_tidu_x == 0] = 1e-8  # 处理 X 方向为 0 的情况
    angle = img_tidu_y / img_tidu_x  # 计算梯度方向

    # 3. 非极大值抑制，保留潜在边缘中的局部最大值
    img_yizhi = np.zeros(img_tidu.shape)  # 初始化非极大值抑制后的图像
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 标记是否抑制
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 获取 3x3 邻域
            #
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
                img_yizhi[i, j] = img_tidu[i, j]  # 保留局部最大值

    # 4. 双阈值检测与边缘连接，确定真正的边缘
    lower_boundary = img_tidu.mean() * low_threshold_ratio  # 低阈值
    high_boundary = lower_boundary * high_threshold_ratio  # 高阈值
    zhan = []  # 边缘追踪栈
    for i in range(1, img_yizhi.shape[0] - 1):
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 高于高阈值，确定为边缘
                img_yizhi[i, j] = 255  # 标记为边缘
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:
                img_yizhi[i, j] = 0  # 低于低阈值，舍弃

    # 使用栈进行边缘追踪
    while zhan:
        temp_1, temp_2 = zhan.pop()
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        for di in range(3):
            for dj in range(3):
                if lower_boundary < a[di, dj] < high_boundary:
                    img_yizhi[temp_1 + di - 1, temp_2 + dj - 1] = 255
                    zhan.append([temp_1 + di - 1, temp_2 + dj - 1])

    # 过滤剩余的非边缘点
    img_yizhi[img_yizhi != 0] = 255

    return img_yizhi.astype(np.uint8)





def warp_perspective_matrix(src, dst):
    if src.shape[0] != dst.shape[0] or src.shape[0] < 4:
        raise ValueError(
            "At least 4 corresponding points are required and src, dst must have the same number of points.")
    if src.shape[1] != 2 or dst.shape[1] != 2:
        raise ValueError("Each point should have 2 coordinates (x, y).")

    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # A*warpMatrix=B
    B = np.zeros((2 * nums, 1))
    # 填充
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]


    # 用np.linalg.inv(A)求出A的逆矩阵，然后用@表示与B矩阵相乘，求出warpMatrix
    warpMatrix = np.linalg.inv(A) @ B
    # 展平结果
    warpMatrix = warpMatrix.flatten()
    # 添加a_33 = 1，并重新reshape成3x3矩阵
    warpMatrix = np.append(warpMatrix, 1.0).reshape((3, 3))
    return warpMatrix



# 调用示例
if __name__ == '__main__':
    # Canny调用
    img_result = canny_edge_detection('lenna.png', sigma=1.52, low_threshold_ratio=0.5, high_threshold_ratio=3.0)
    plt.imshow(img_result, cmap='gray')
    plt.axis('off')

    # 透视变换调用
    img = cv2.imread('photo1.jpg')
    result3 = img.copy()
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    warpMatrix = warp_perspective_matrix(src, dst)
    result = cv2.warpPerspective(result3, warpMatrix, (337, 488))
    cv2.imshow("src", img)
    cv2.imshow("result", result)

    plt.show()
    cv2.waitKey(0)