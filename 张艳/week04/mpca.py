import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris
import pandas as pd

# 鸢尾花数据集，用sklearn库的PCA算法

# 加载鸢尾花数据集
x, y = load_iris(return_X_y=True)

# 创建PCA对象，设置降维后主成分数目为2
pca = dp.PCA(n_components=2)

# 执行pca算法，返回降维后的二维数据
reduced_x = pca.fit_transform(x)

# 输出该数组的xlsx文件
# data=pd.DataFrame(reduced_x)
# data['target'] = y
# data.to_excel('reduced_x.xlsx',index=False)

# 根据y值把reduced_x分为三类一维数组
red_x, red_y = [], []
green_x, green_y = [], []
blue_x, blue_y = [], []
for i in range(len(reduced_x)):
    if y[i]==0 :
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i]==1:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
    else:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])

# 在一个散点图上 绘制这三类一维数组
plt.title("sklearn的PCA算法")
plt.scatter(red_x,red_y,c='r',marker='X')
plt.scatter(green_x,green_y,c='g',marker='D')
plt.scatter(blue_x,blue_y,c='b',marker='.')
plt.show()

'''
import pandas as pd
from sklearn.datasets import load_iris

# 加载鸢尾花数据集
x, y = load_iris(return_X_y=True)

# 创建DataFrame
data = pd.DataFrame(x, columns=load_iris().feature_names)
data['target'] = y  # 添加1列 列名为target

# 把 鸢尾花数据集 保存到Excel文件
data.to_excel('iris_data.xlsx', index=False)  # index=True
'''
