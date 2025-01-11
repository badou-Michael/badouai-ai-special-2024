import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn import datasets


data = datasets.load_iris()
#打印鸟类的属性名
print(data.feature_names)
#加载数据，x表示数据，y表示分类标签
x,y=datasets.load_iris(return_X_y=True)
#降维 缺点 无法人工选取特定的特征向量
pca = dp.PCA(n_components=3)
reduced_x = pca.fit_transform(x)



red_x,red_y,red_z=[],[],[]
blue_x,blue_y,blue_z=[],[],[]
green_x,green_y,green_z=[],[],[]

# #按鸢尾花的类别将降维后的数据点保存在不同的表中
for i in range(len(reduced_x)):
    # y为数据的分类标签 0 1 2代表不同类型的鸟类，数据集变为两维后，0维代表X轴，1维代表Y轴 2维代表Z轴
    # 通过三个维度查看分类的效果
        if y[i]==0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
            red_z.append(reduced_x[i][2])
        elif y[i]==1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
            blue_z.append(reduced_x[i][2])
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])
            green_z.append(reduced_x[i][2])

ax = plt.subplot(111, projection='3d') # 创建一个三维的绘图工程
# 将数据点分成三部分画，在颜色上有区分度
ax.scatter(red_x, red_y,red_z, c='r',marker='x') # 绘制数据点
ax.scatter(blue_x, blue_y, blue_z, c='g',marker='D')
ax.scatter(green_x, green_y, green_z, c='b',marker='.')

ax.set_zlabel('Z') # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
#在三维中显示数据集分布
#从三维中 俯视 XY轴 XZ轴  均能被良好分类  YZ维度无法较好分类  可见特征值大小决定了样本集属性在样本中的重要程度
plt.show()