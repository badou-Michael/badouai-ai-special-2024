import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


class PCAImplementation:
    def __init__(self, n_components):
        self.n_components = n_components
        self.eigenvectors = None
        self.eigenvalues = None

    def fit(self, X):
        # 去中心化
        X_centered = X - np.mean(X, axis=0)

        # 计算协方差矩阵
        C = np.dot(X_centered.T, X_centered) / X.shape[0]

        # 计算特征值和特征向量
        self.eigenvalues, self.eigenvectors = np.linalg.eig(C)

        # 按特征值从大到小排序特征向量
        ind = np.argsort(-self.eigenvalues)
        self.eigenvectors = self.eigenvectors[:, ind[:self.n_components]]

    def transform(self, X):
        # 将数据投影到主成分上
        X_centered = X - np.mean(X, axis=0)
        return np.dot(X_centered, self.eigenvectors)


def main():
    # 加载 Iris 数据集
    iris = load_iris()
    X = iris.data  # 特征矩阵
    K = 2  # 降维到2维

    # 实例化 PCA 类
    pca = PCAImplementation(n_components=K)

    # 拟合数据并转换
    pca.fit(X)
    X_transformed = pca.transform(X)

    # 输出结果
    print("降维后的数据：")
    print(X_transformed)
    print("原始数据：")
    print(X)

    # 可视化降维结果
    plt.figure(figsize=(8, 6))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=iris.target, cmap='viridis')
    plt.xlabel('主成分 1')
    plt.ylabel('主成分 2')
    plt.title('PCA 降维结果')
    plt.colorbar(label='种类')
    plt.show()


if __name__ == "__main__":
    main()
