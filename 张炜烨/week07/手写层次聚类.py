import numpy as np

# 计算欧几里得距离
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# 初始化距离矩阵
def create_distance_matrix(data):
    n = len(data)
    distance_matrix = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            distance = euclidean_distance(data[i], data[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    return distance_matrix

# 合并簇
def merge_clusters(clusters, c1, c2):
    new_cluster = clusters[c1] + clusters[c2]
    clusters.pop(max(c1, c2))
    clusters.pop(min(c1, c2))
    clusters.append(new_cluster)

# 层次聚类算法
def hierarchical_clustering(data):
    clusters = [[i] for i in range(len(data))]
    distance_matrix = create_distance_matrix(data)
    
    while len(clusters) > 1:
        # 找到最小距离的簇对
        min_dist = np.inf
        c1, c2 = -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                for p1 in clusters[i]:
                    for p2 in clusters[j]:
                        if distance_matrix[p1][p2] < min_dist:
                            min_dist = distance_matrix[p1][p2]
                            c1, c2 = i, j
        
        # 合并最小距离的簇对
        merge_clusters(clusters, c1, c2)
        print(f"合并簇: {c1} 和 {c2}")

    return clusters

# 示例数据
data = [
    [1, 2],
    [2, 3],
    [5, 5],
    [8, 8]
]

# 执行层次聚类
result = hierarchical_clustering(data)
print("最终簇：", result)