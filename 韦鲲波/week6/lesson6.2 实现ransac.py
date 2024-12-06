import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
========================================================================================================================
最小二乘法（Least Square Method）
'''

def lsm(X):
    s11 = s12 = s13 = s21 = 0

    # X = np.array(
    #     [
    #         [0, 0], [1, 1], [2, 3], [5, 4], [9, 8],
    #     ]
    # )
    # print(X.shape)

    for i in range(X.shape[0]):
        s11 += X[i][0] * X[i][1]  # xy的和
        s12 += X[i][0]  # x和
        s13 += X[i][1]  # y和
        s21 += X[i][0] ** 2  # x方的和

    k1 = X.shape[0] * s11 - s12 * s13
    k2 = X.shape[0] * s21 - s12 ** 2

    k = k1 / k2
    b = (s13 - k * s12) / X.shape[0]
    # print(f'y = {k:.2f}x {b:.2f}')

    return k, b




'''
========================================================================================================================
RANSAC
随机采样一致性
random sample consensus
'''

np.random.seed()
# 创建初始化数据集 answer x y
ax = 20 * np.random.random(size=(500, 1))
ak = 50 * np.random.normal()
ay = np.dot(ax, ak)
# print(len(ax))

# plt.scatter(ax, ay)
# plt.show()

# 将数据集的点稍微进行一些离散，增加一些测试难度 text x y
tx = ax + 2 * np.random.random(size=ax.shape)
ty = ay + 2 * np.random.normal(size=ay.shape)
# print(tx[0])
# print(ty[0])
# print(len(tx))

# plt.scatter(tx, ty)
# plt.show()

# 求k和b
tp = np.column_stack((tx, ty))  # 合并tx ty
kz, bz = lsm(tp)  # 用上面的最小二乘法求k和b
y1 = kz * min(tx) + bz
y2 = kz * max(tx) + bz

# 输出线段
plt.plot([min(tx), max(tx)], [y1, y2], color='orange')

# 再加入一些更离群的一些点
n = 250

# 把这些点放到测试集中
minpoint = np.array([np.random.randint(0, 500) for _ in range(n)])
# print(minpoint)
# print(type(minpoint))

tx[minpoint] = 20 * np.random.random(size=(n, 1))
ty[minpoint] = 400 * np.random.normal(size=(n, 1)) +600
# print(tx[minpoint])

plt.scatter(tx, ty)



# 编辑RANSAC的程序
def ransac(n, K, T, thre):
    # n作为每次循环选取进行尝试的点的数量n
    # 最大循环次数停止阈值K
    # 符合数量的阈值T，记录num符合数量时的k和b
    # 传入thre作为原y和计算出来的y的差值阈值
    p = np.column_stack((tx, ty))

    tar = []  # 符合要求的num值及其对应的k和b
    loop = 0  # loop 循环次数
    conv = []

    while loop <= K:
        num = 0  # 内群数量
        # 打乱数组，截取前n个
        calc = np.random.permutation(p)
        k, b = lsm(calc[:n])

        # 计算外群符合的点的数量
        for i in calc[n:]:
            # print('i', i)
            y = k * i[0] + b
            yz = kz * i[0] + bz
            if abs(yz - y) <= thre:
                num += 1
        # print('num', num)

        # 符合给定数量T的，num k b都存入tar变量
        if num >= T:
            tar.append([num, k, b])

        # 对loop与num的值做一个记录
        conv.append([loop, num])

        loop += 1

    # 对conv的顺序进行优化
    conv = np.array(conv)
    conv_index = conv[:, 1].argsort()
    conv_sorted = conv[conv_index]
    mx = np.reshape(
        [i for i in range(conv_sorted.shape[0])],
        (conv_sorted.shape[0], 1),
    )
    # print(mx.shape)
    my = np.reshape(
        conv_sorted[:, 1],
        (conv_sorted.shape[0], 1)
    )
    # print(my.shape)
    merge = np.hstack((mx, my))
    print(merge)

    tar = np.array(tar)
    # 判断tar是否为空
    if len(tar) > 0:
        # print(tar)
        max_num = np.max(tar[:, 0])
        max_index = np.where(tar[:, 0] == max_num)
        tar_max = tar[max_index, :]

        # 判断tar里是否只有一组数
        if len(tar_max[0]) > 1:
            print('more than one tar')
            print(tar_max)
            # 把所有组的线都放图里
            for i in tar_max:
                k = i[0][1]
                b = i[0][2]
                plt.figure(1)
                plt.plot([min(tx), max(tx)], [k * min(tx) + b, k * max(tx) + b], color='r')
        else:  # 只有一组就放一个图
            print('only one tar')
            print(tar_max)
            k = tar_max[0][0][1]
            b = tar_max[0][0][2]
            plt.figure(1)
            plt.plot([min(tx), max(tx)], [k * min(tx) + b, k * max(tx) + b], color='r')

    else:
        print('empty tar')

    plt.figure(2)
    plt.grid(True)
    plt.xlabel('loop')
    plt.ylabel('num')
    plt.scatter(conv[:, 0], conv[:, 1])

    plt.figure(3)
    plt.grid(True)
    plt.xlabel('loop')
    plt.ylabel('num')
    plt.scatter(merge[:, 0], merge[:, 1])



q = ransac(5, 1000, 100, 5)

plt.show()

'''
========================================================================================================================
'''

















