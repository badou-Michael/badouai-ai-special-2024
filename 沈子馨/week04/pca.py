from sklearn.datasets.base import load_iris
import matplotlib.pyplot as plt
import sklearn.decomposition as dp

x, y = load_iris(return_X_y=True)
pca = dp.PCA(n_components=2)
reduced_x = pca.fit_transform(x)

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(len(reduced_x)): #按鸢尾花的类别将降维后的数据点保存在不同的表中
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, lue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()