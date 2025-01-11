from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

data, target = load_iris(return_X_y=True)
# 调用PCA主成分分析，n_components代表着输出的维度
pca = PCA(n_components=2)
# 对数据进行拟合和转换
X_transformed = pca.fit_transform(data)
# 通过特征值计算得到的主成分所占的半分比
explained_variance_ratio = pca.explained_variance_ratio_
# 数据
print("原始数据：\n", data)
print("降维的数据：\n", X_transformed)
print("降维后的主成分所占的百分比：\n", explained_variance_ratio)

