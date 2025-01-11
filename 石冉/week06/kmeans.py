from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

X = [
    [1.0, 1.0],
    [1.5, 1.5],
    [2.0, 2.0],
    [6.0, 5.0],
    [7.0, 6.0],
    [8.0, 7.0],
    [1.0, 0.5],
    [1.5, 0.5],
    [2.0, 0.0],
    [6.0, 4.0],
    [7.0, 5.0],
    [8.0, 6.0]
]

#引用kmeans进行聚类，目标是聚2类
cl=KMeans(n_clusters=2)
y_predict=cl.fit_predict(X)
#输出聚类函数和预测结果
print(cl)
print('y_predict=',y_predict)

#将结果可视化
x=[n[0] for n in X] #x轴
y=[n[1] for n in X] #y轴
plt.scatter(x,y,c=y_predict,marker='*')
#标题
plt.title('max and min')
plt.show()
