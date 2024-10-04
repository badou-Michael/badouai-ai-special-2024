from matplotlib import pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris

# 加载莺尾花数据集，x表示数据集中的属性数据：特征矩阵，y表示数据标签:目标向量
# return_X_y：如果设置为 True，则返回 (data, target) 两个数组；如果设置为 False（默认值），则返回一个 Bunch 对象，包含多个字段。
x, y = load_iris(return_X_y=True)
print(f'x={x}, y={y}')
# 加载pca算法，设置降维后主成分数目为2
pca = dp.PCA(n_components=2)
# 对原始数据进行降维，保存在reduced_x中
reduced_x = pca.fit_transform(x)


red_x, red_y=[],[]
blue_x, blue_y=[],[]
green_x, green_y=[],[]
for i in range(len(reduced_x)): #按鸢尾花的类别将降维后的数据点保存在不同的表中
    if y[i]==0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i]==1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()