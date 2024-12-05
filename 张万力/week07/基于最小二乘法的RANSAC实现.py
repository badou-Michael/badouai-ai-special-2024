import numpy as np
import matplotlib.pyplot as plt


class RANSAC:
    def __init__(self,
                 min_samples=2,  # 最小样本数
                 residual_threshold=1.0,  # 残差阈值
                 max_iterations=100):  # 最大迭代次数
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_iterations = max_iterations

        # 存储模型参数
        self.best_model = None
        self.best_inliers = None

    def least_squares_fit(self, X, y):
        """
        最小二乘法拟合直线模型
        y = kx + b
        返回斜率k和截距b
        """
        # 构造设计矩阵
        A = np.vstack([X, np.ones(len(X))]).T

        # 最小二乘求解
        k, b = np.linalg.lstsq(A, y, rcond=None)[0]

        return k, b

    def calculate_residuals(self, X, y, k, b):
        """
        计算残差
        """
        # 预测值
        y_pred = k * X + b

        # 计算残差绝对值
        return np.abs(y - y_pred)

    def fit(self, X, y):
        """
        RANSAC主算法
        """
        n_samples = len(X)
        best_inliers_count = 0

        for _ in range(self.max_iterations):
            # 1. 随机抽样
            sample_indices = np.random.choice(
                n_samples,
                self.min_samples,
                replace=False
            )

            # 2. 使用最小样本拟合模型
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            # 最小二乘拟合
            k, b = self.least_squares_fit(X_sample, y_sample)

            # 3. 计算残差
            residuals = self.calculate_residuals(X, y, k, b)

            # 4. 判断内点
            inliers_mask = residuals < self.residual_threshold
            inliers_count = np.sum(inliers_mask)

            # 5. 更新最佳模型
            if inliers_count > best_inliers_count:
                best_inliers_count = inliers_count
                self.best_model = (k, b)
                self.best_inliers = inliers_mask

        # 6. 使用所有内点重新拟合模型
        if self.best_model is not None:
            X_inliers = X[self.best_inliers]
            y_inliers = y[self.best_inliers]
            self.best_model = self.least_squares_fit(X_inliers, y_inliers)

        return self


def generate_noisy_data():
    """
    生成带噪声的数据
    """
    # 设置随机种子
    np.random.seed(42)

    # 生成自变量
    X = np.linspace(0, 10, 100)

    # 真实模型参数
    true_k = 2  # 斜率
    true_b = 1  # 截距

    # 生成真实值
    y_true = true_k * X + true_b

    # 添加高斯噪声
    noise = np.random.normal(0, 1, X.shape)
    y = y_true + noise

    # 添加离群点
    outlier_indices = np.random.choice(len(X), 10, replace=False)
    y[outlier_indices] += np.random.uniform(-10, 10, len(outlier_indices))

    return X, y


def main():
    # 生成数据
    X, y = generate_noisy_data()

    # 普通最小二乘拟合
    A = np.vstack([X, np.ones(len(X))]).T
    k_ols, b_ols = np.linalg.lstsq(A, y, rcond=None)[0]

    # RANSAC拟合
    ransac = RANSAC(
        min_samples=2,
        residual_threshold=2.0,
        max_iterations=100
    )
    ransac.fit(X, y)

    # 可视化结果
    plt.figure(figsize=(10, 6))

    # 原始数据散点
    plt.scatter(X, y, c='blue', alpha=0.5, label='Original Data')

    # 普通最小二乘直线
    plt.plot(X, k_ols * X + b_ols,
             color='green', label='OLS Regression')

    # RANSAC直线
    k_ransac, b_ransac = ransac.best_model
    plt.plot(X, k_ransac * X + b_ransac,
             color='red', label='RANSAC Regression')

    # 标记内点和外点
    plt.scatter(X[ransac.best_inliers],
                y[ransac.best_inliers],
                c='green', label='Inliers')
    plt.scatter(X[~ransac.best_inliers],
                y[~ransac.best_inliers],
                c='red', label='Outliers')

    plt.title('RANSAC vs OLS Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # 打印模型参数
    print("OLS Model: y = {:.2f}x + {:.2f}".format(k_ols, b_ols))
    print("RANSAC Model: y = {:.2f}x + {:.2f}".format(k_ransac, b_ransac))


# 运行主函数
if __name__ == '__main__':
    main()
