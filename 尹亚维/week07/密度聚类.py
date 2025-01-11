from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
# 表示我们只取特征空间中的4个维度,（萼片长度、萼片宽度、花瓣长度、花瓣宽度）
X = iris.data[:, :4]
print(f"X={X}, X.shape ={X.shape}")

# 对数据进行聚类，并获取聚类标签，
# eps是一个距离阈值，用于确定邻域内的点是否被认为是“近邻”
# min_samples是一个整数，表示一个点的邻域内至少需要有多少个点才能使该点成为核心点
dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X)
label_pred = dbscan.labels_

# 绘制结果
unique_labels = set(label_pred)
colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta']
markers = ['o', '*', '+', 'x', '^', 's', 'D']

for label, color, marker in zip(unique_labels, colors, markers):
    if label == -1:
        # 噪声点
        plt.scatter(X[label_pred == label, 0], X[label_pred == label, 1], c=color, marker=marker, label='noise')
    else:
        plt.scatter(X[label_pred == label, 0], X[label_pred == label, 1], c=color, marker=marker, label=f'label{label}')

plt.xlabel('sepal length')  # 萼片的长度
plt.ylabel('sepal width')  # 萼片的宽度
plt.legend(loc=2)
plt.show()

# 绘制结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')  # 萼片的长度
plt.ylabel('sepal width')  # 萼片的宽度
plt.legend(loc=2)
plt.show()