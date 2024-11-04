
实现透视变换和K-Means

import numpy as np
import cv2


def warp_perspective_matrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4 # assert断言语句

    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # A*warp_matrix=B
    B = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    A = np.mat(A) # 将A矩阵转换为NumPy的矩阵类型

    # 用A.I求出A的逆矩阵，然后与B相乘，求出warp_matrix
    warp_matrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32

    # 之后为结果的后处理
    warp_matrix = np.array(warp_matrix).T[0]
    warp_matrix = np.insert(warp_matrix, warp_matrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warp_matrix = warp_matrix.reshape((3, 3))
    return warp_matrix


# if __name__ == '__main__':
#     print('warp_matrix')
#     src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
#     src = np.array(src)
#
#     dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
#     dst = np.array(dst)
#
#     warp_matrix = warp_perspective_matrix(src, dst)
#     print(warp_matrix)


img = cv2.imread('photo1.jpg')

result3 = img.copy()
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (337, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)


k-means

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# 读取原始图像灰度颜色
img = cv2.imread('../week02/lenna.png', 0)

# 获取图像高度、宽度
rows, cols = img.shape[:]

# 图像二维像素转换为一维
data = img.reshape((rows * cols, 1))
data = np.float32(data)

# 停止条件 (type,max_iter,epsilon)
# cv2.TERM_CRITERIA_EPS：当迭代之间的变化小于指定的精度 epsilon 时停止迭代。
# cv2.TERM_CRITERIA_MAX_ITER：当达到最大迭代次数时停止迭代。
# 10：最大迭代次数。
# 1.0：精度阈值 epsilon
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means聚类 聚集成4类
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

# 生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
# titles = [u'原始图像', u'聚类图像']
# images = [img, dst]
# for i in range(2):
#     plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'),
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()


# 读取原始图像
img = cv2.imread('../week02/lenna.png')
print(img.shape)

# 图像二维像素转换为一维
data = img.reshape((-1, 3))
data = np.float32(data)

# 停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means聚类 聚集成2类
compactness, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)

# K-Means聚类 聚集成4类
compactness, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)

# K-Means聚类 聚集成8类
compactness, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)

# K-Means聚类 聚集成16类
compactness, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)

# K-Means聚类 聚集成64类
compactness, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)

# 图像转换回uint8二维类型
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst2 = res.reshape((img.shape))

centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]
dst4 = res.reshape((img.shape))

centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape((img.shape))

centers16 = np.uint8(centers16)
res = centers16[labels16.flatten()]
dst16 = res.reshape((img.shape))

centers64 = np.uint8(centers64)
res = centers64[labels64.flatten()]
dst64 = res.reshape((img.shape))

# 图像转换为RGB显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4',
          u'聚类图像 K=8', u'聚类图像 K=16', u'聚类图像 K=64']
images = [img, dst2, dst4, dst8, dst16, dst64]
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray'),
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()



X = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]
     ]

# 输出数据集
print(X)


# 第二部分：KMeans聚类
# clf = KMeans(n_clusters=3) 表示类簇数为3，聚成3类数据，clf即赋值为KMeans
clf = KMeans(n_clusters=3)
y_pred = clf.fit_predict(X)

# 输出完整Kmeans函数，包括很多省略参数
print(clf)
# 输出聚类预测结果
print("y_pred = ", y_pred)


# 可视化绘图


# 获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列
x = [n[0] for n in X]
print(x)
y = [n[1] for n in X]
print(y)

 # 绘制散点图 参数：x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;

plt.scatter(x, y, c=y_pred, marker='x')

# 绘制标题
plt.title("Kmeans-Basketball Data")

# 绘制x轴和y轴坐标
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")

# 设置右上角图例
plt.legend(["A", "B", "C"])

# 显示图形
plt.show()
