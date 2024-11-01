import numpy as np

def compute_perspective_transform(src_points, dst_points):
    # 确保源点和目标点数量一致，且至少有4个点
    assert src_points.shape[0] == dst_points.shape[0] >= 4

    A = []
    B = []

    for i in range(src_points.shape[0]):
        x_src, y_src = src_points[i]
        x_dst, y_dst = dst_points[i]

        # 构建矩阵A的两行
        A.append([x_src, y_src, 1, 0, 0, 0, -x_src*x_dst, -y_src*x_dst])
        A.append([0, 0, 0, x_src, y_src, 1, -x_src*y_dst, -y_src*y_dst])

        # 构建向量B
        B.append(x_dst)
        B.append(y_dst)

    # 将A和B转换为NumPy数组
    A = np.array(A)
    B = np.array(B)

    # 使用最小二乘法求解方程 Ah = B，得到透视变换矩阵的参数h
    h = np.linalg.lstsq(A, B, rcond=None)[0]

    # 将参数h转换为3x3矩阵，并在最后补上元素1
    h = np.append(h, 1).reshape((3, 3))
    return h

if __name__ == '__main__':
    print('计算透视变换矩阵:')
    # 定义源点坐标
    src = np.array([
        [10.0, 457.0],
        [395.0, 291.0],
        [624.0, 291.0],
        [1000.0, 457.0]
    ])

    # 定义目标点坐标
    dst = np.array([
        [46.0, 920.0],
        [46.0, 100.0],
        [600.0, 100.0],
        [600.0, 920.0]
    ])

    # 计算并输出透视变换矩阵
    warp_matrix = compute_perspective_transform(src, dst)
    print('透视变换矩阵为：')
    print(warp_matrix)