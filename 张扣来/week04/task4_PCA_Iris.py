import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris

from task.实现PCA04 import reduced_x

# 加载鸢尾花数据，x表示数据集中的属性数据，y表示数据标签
x, y = load_iris(return_X_y=True)
pca = dp.PCA(n_components=2)#加载pca算法，处理到2阶矩阵
reduced_x = pca.transform(x)#降维处理

red_x,red_y = [],[]
green_x,green_y = [],[]
blue_x,blue_y = [],[]
for i in range(len(reduced_x)):
    # 当数据标签为0的时候分为红色,y为1分为绿色，其余为蓝色
    if [y] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    if [y] ==1:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][0])
    else:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][0])
# plt.scatter散点图绘制，不同颜色
plt.scatter(red_x,red_y,c="r",marker="s")
plt.scatter(green_x,green_y,c="g",marker="o")
plt.scatter(blue_x,blue_y,c="b",marker="h")
plt.show()