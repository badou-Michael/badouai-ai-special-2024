import numpy as np
import random

def ransac(data, model_func, n_samples, threshold, max_iterations=1000, min_inliers=10):
    best_model = None
    best_inliers = []
    best_error = float('inf')

    for _ in range(max_iterations):
        # Step 1: 随机抽取样本
        sample_indices = random.sample(range(len(data)), n_samples)
        sample_points = data[sample_indices]

        # Step 2: 通过样本拟合模型
        model = model_func(sample_points)

        # Step 3: 计算内点集
        inliers = []
        for i, point in enumerate(data):
            if i not in sample_indices:
                error = model_error(model, point)
                if error < threshold:
                    inliers.append(i)

        # Step 4: 记录最佳模型
        if len(inliers) >= min_inliers:
            total_error = sum(model_error(model, data[i]) for i in inliers)
            if total_error < best_error:
                best_model = model
                best_inliers = inliers
                best_error = total_error

    return best_model, best_inliers

def model_func(points):
    # 用最小二乘法拟合线性模型（例如 y = ax + b）
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b

def model_error(model, point):
    a, b = model
    x, y = point
    y_pred = a * x + b
    return abs(y - y_pred)

# 示例数据
data = np.random.rand(100, 2) * 100  # 生成一些随机数据

# 执行 RANSAC
best_model, best_inliers = ransac(data, model_func, n_samples=2, threshold=5)
print("最佳模型参数:", best_model)
print("内点数量:", len(best_inliers))
