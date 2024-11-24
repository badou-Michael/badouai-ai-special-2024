import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets.base import load_iris
'''
load_iris函数用于加载著名的Iris数据集。
Iris数据集是一个经典的机器学习数据集
包含150个样本,每个样本有4个特征（花萼长度、花萼宽度、花瓣长度、花瓣宽度）
以及一个目标变量（花的种类，有三种：Iris-setosa、Iris-versicolor、Iris-virginica）
'''

# 加载数据，x表示数据集中的属性数据，y表示数据标签
'''
x 是特征矩阵，每一行代表一个样本，每一列代表一个特征。
    例如，如果鸢尾花数据集有 150 个样本和 4 个特征，那么 x 将是一个 150x4 的矩阵。
y 是目标向量，每一行代表一个样本的类别标签。
    例如，如果鸢尾花数据集有 150 个样本，那么 y 将是一个 150x1 的向量。
    y[0],y[1],y[2]各表示一种花的种类
'''
x,y = load_iris(return_X_y=True)
# 加载pca算法，设置降维后主成分数目为2
pca = dp.PCA(n_components=2)
# 对原始数据进行降维
reduced_x = pca.fit_transform(x)


red_x, red_y = [],[]
blue_x, blue_y = [],[]
green_x, green_y = [],[]
# 按照鸢尾花的类别将降维后的数据点保存在不同表中
for i in range(len(reduced_x)):
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])


plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='d')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()




