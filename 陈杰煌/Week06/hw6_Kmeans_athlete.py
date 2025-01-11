import os
os.environ["OMP_NUM_THREADS"] = "1" 
#  Windows 上运行 scikit-learn 的 KMeans 算法时，使用了 Intel 的数学核心库（MKL）。
# 当线程数超过可用的计算块数时，可能会导致内存泄漏。为了解决这个问题，您可以将环境变量 OMP_NUM_THREADS 设置为 1。

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
 
"""
第一部分：数据集
X表示二维矩阵数据，篮球运动员比赛数据
总共20行，每行两列数据
第一列表示球员每分钟助攻数：assists_per_minute
第二列表示球员每分钟得分数：points_per_minute
"""
X = [
    [0.0888, 0.5885],
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

"""
第二部分：KMeans聚类
"""
# 创建KMeans模型并进行聚类
kmeans = KMeans(n_clusters=3)
y_pred = kmeans.fit_predict(X) # fit_predict(X)可以看做是fit(X)和predict(X)的结合 作用是训练数据X并对X的类进行预测

# 输出聚类结果
print("聚类结果:", y_pred)

"""
第三部分：可视化绘图
"""
# 提取数据的两列
x = [point[0] for point in X]
y = [point[1] for point in X]

# 绘制散点图
plt.scatter(x, y, c=y_pred, marker='x')

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 设置标题和轴标签
plt.title("KMeans-篮球数据聚类")
plt.xlabel("每分钟助攻数")
plt.ylabel("每分钟得分数")

# 显示图形
plt.show()