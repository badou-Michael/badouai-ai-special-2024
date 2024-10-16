import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris

'''
x：加载到的数据集的属性数据（即每个样本有4个特征）
y：对应的数据标签（0,1,2 分别表示鸢尾花的3个类别）
'''
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape)  # (150,4) (150,)

pca = dp.PCA(n_components=2)  # 加载PCA算法，设置降维的主成分为2
result_x = pca.fit_transform(x)  # 对原始数据进行降维

print(result_x)

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

# 按照鸢尾花的类别将降维后的数据点保存的不同的表中
for i in range(len(result_x)):
    if y[i] == 0:
        red_x.append(result_x[i, 0])
        red_y.append(result_x[i, 0])
    elif y[i] == 1:
        blue_x.append(result_x[i, 0])
        blue_y.append(result_x[i, 0])
    else:
        green_x.append(result_x[i, 0])
        green_y.append(result_x[i, 0])

# 绘制3种类别鸢尾花数据点的散点图
plt.scatter(red_x,red_y,c='r',marker='x')  # marker='x' 指定点的形状为叉形
plt.scatter(blue_x,blue_y,c='b',marker='D') # 标记形状为菱形 (marker='D')
plt.scatter(green_x,green_y,c='g',marker='.') # 标记形状为圆点 (marker='.')
plt.show()


