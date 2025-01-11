import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import random

#Hierarchical Clustering
X,Y = make_blobs(n_samples=300,centers=4,cluster_std=0.60,random_state=0)
clustering = AgglomerativeClustering(n_clusters=4).fit(X)
cluster_labels = clustering.labels_
plt.figure(figsize=(8,6))
plt.scatter(X[:,0],X[:,1],c=cluster_labels,cmap='viridis',s=50,alpha=0.7)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Clustering')
plt.show()

#ransac
np.random.seed(0)
n_samples = 100
n_outliers = 10
X = np.random.rand(n_samples,1)*100
Y = 2.5 * X + 1.2 + np.random.normal(size = (n_samples,1))
#切片操作
X[:n_outliers] = np.random.rand(n_outliers,1)*100
Y[:n_outliers] = np.random.rand(n_outliers,1)*100

def ransac(X,Y,n_iterations,threshold,min_samples):
    best_model = None
    best_inliers = []
    best_error = np.inf
    for _ in range(n_iterations):
        sample_indices = random.sample(range(len(X)),min_samples)      
        #从0到len(X) - 1这个索引范围内随机抽取min_samples个不重复的索引值，并将这些索引值组成的列表赋值给sample_indices
        X_sample = X[sample_indices]
        Y_sample = Y[sample_indices]
        model = np.polyfit(X_sample.flatten(),Y_sample.flatten(),1)
        #flatten将多维数组转换为一维数组
        #polyfit多项式拟合函数，自变量、因变量、拟合次数（1即进行一次多项式拟合）
        errors = np.abs(Y-(model[0]*X+model[1]))
        #model[0]是斜率m，model[1]是截距b
        inliers = np.where(errors<threshold)[0]
        #where 条件筛选函数
        #这些索引以元组的形式返回，如果condition是一维数组，那么返回的元组中只有一个元素（这个元素是一个一维索引数组）；
        #如果condition是多维数组，返回的元组中元素个数与condition的维度数相同，每个元素都是对应维度上满足条件的索引数组
        #[0] 只取元组中的第一个元素（索引数组）
        error = np.sum(errors)
        if len(inliers)>len(best_inliers) or (len(inliers)==len(best_inliers) and error < best_error):
            best_model = model
            best_inliers = inliers
            best_error = error
    return best_model,best_inliers

n_iterations = 100
threshold = 5
min_samples = 2
best_model,best_inliers = ransac(X,Y,n_iterations,threshold,min_samples)

plt.scatter(X,Y,color='b',label ='Data points')
plt.scatter(X[best_inliers],Y[best_inliers],color='r',label='inliers')
plt.plot(X,best_model[0] * X + best_model[1],color='g',label='RANSAC model')
plt.legend()
plt.show()
