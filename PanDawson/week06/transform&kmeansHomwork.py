import cv2
import numpy as np


def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    
    nums = src.shape[0]
    A = np.zeros((2*nums, 8)) # A*warpMatrix=B
    B = np.zeros((2*nums, 1))
    for i in range(0, nums):
        A_i = src[i,:]
        B_i = dst[i,:]
        A[2*i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                       -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        B[2*i] = B_i[0]
        
        A[2*i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                       -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[2*i+1] = B_i[1]
 
    A = np.mat(A)
    #用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B #求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    
    #之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0) #插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix
 
dst = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
src = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
warpMatrix = WarpPerspectiveMatrix(src, dst)
print(warpMatrix)

height,width,channels = img.shape
transformed_img = np.zeros((height, width, img.shape[2]), dtype=img.dtype)
transformed_img.shape

def bilinear_interpolation(img, x, y):
    """
    双线性插值函数
    """
    x0 = int(x)
    y0 = int(y)
    x1 = x0+1;
    y1 = y0+1;

    if x1 >= img.shape[1]:
        x1 = img.shape[1] - 1
    if y1 >= img.shape[0]:
        y1 = img.shape[0] - 1

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id

for y in range(height-1):
    for x in range(width-1):
        # 将原始图像的坐标转换为齐次坐标形式 [x, y, 1]
        original_coord = np.array([x, y, 1])

        # 应用透视变换矩阵
        transformed_coord = np.dot(warpMatrix, original_coord)

        # 将齐次坐标转换回二维坐标
        transformed_x = transformed_coord[0] / transformed_coord[2]
        transformed_y = transformed_coord[1] / transformed_coord[2]


        # 检查变换后的坐标是否在目标图像范围内
        if 0 <= transformed_x < width and 0 <= transformed_y < height:
            # 将原始图像中的像素值赋给目标图像中对应的位置
            transformed_img[y, x] = bilinear_interpolation(img, transformed_x, transformed_y)

cv2.imshow("original img",img)
cv2.imshow("after transformed img",transformed_img)
cropped_image = transformed_img[0:488, 0:377]
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)

class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
    
    def fit(self, X):
        n_samples, n_features = X.shape
    
        # 初始化聚类中心
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        #生成随机样本,从n_samples这么多个样本中随机选择出n_clusters个样本出来，作为初始聚类中心，存放在centroids中
        #replace=False参数表示抽样是无放回的。也就是说，每个样本在一次抽样过程中只能被选中一次
        
        for _ in range(self.max_iter):
            # 计算每个样本到各个聚类中心的距离
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            #X是数据集，形状为(n_samples, n_features)
            #self.centroids是聚类中心，形状为(n_clusters, n_features)
            #通过[:, np.newaxis]操作，将self.centroids的维度从(n_clusters, n_features)变为(n_clusters, 1, n_features)
            #减法操作是在每个特征维度上进行的，对于X中的每个样本，都会减去所有的聚类中心，
            #得到一个形状为(n_clusters, n_samples, n_features)的数组，表示每个样本与每个聚类中心在每个特征维度上的差值。
            #沿着新增加的第三个维度（axis = 2）进行求和。这个操作将每个样本与每个聚类中心在所有特征维度上的差值平方和计算出来，得到一个形状为(n_clusters, n_samples)的数组。
            #此时，对于每个样本和每个聚类中心，这个值表示它们之间的平方距离
            #对每个样本与每个聚类中心之间的平方距离进行开方操作，得到欧几里得距离。
            #最终distances的形状是(n_clusters, n_samples)，distances[i][j]表示第j个样本到第i个聚类中心的欧几里得距离。
            
            # 分配每个样本到最近的聚类中心
            labels = np.argmin(distances, axis=0)
            #distances是一个二维数组，其形状为(n_clusters, n_samples)
            #argmin: 返回给定数组中最小值的索引
            #当axis = 0时，表示沿着第 0 轴（即垂直方向，对于(n_clusters, n_samples)形状的数组，就是按列方向）进行操作
            #就是找每个样本其到聚类中心的最小值
            
            # 更新聚类中心
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            #labels是一个长度为n_samples的数组
            #labels == i会生成一个布尔型数组，其长度与labels相同，当labels中的元素等于i时，对应位置为True，否则为False。这个布尔型数组用于对数据集X进行索引
            #X是形状为(n_samples, n_features)的数组
            #对属于第i个聚类的所有样本在特征维度上求均值。axis = 0表示沿着第一个维度（即样本维度）进行计算，
            #这样得到的结果是一个形状为(n_features,)的数组，表示第i个聚类的新的聚类中心在各个特征维度上的均值。
            
            
            
            # 如果聚类中心不再变化，则停止迭代
            if np.all(self.centroids == new_centroids):
                break
        
        self.centroids = new_centroids
    
    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        return labels

# 使用示例
if __name__ == "__main__":

    X = np.array([[0.0888, 0.5885],
         [0.1399, 0.8291],
         [0.0747, 0.4974],
         [0.0983, 0.5772],
         [0.1276, 0.5703],
         [0.1671, 0.5835],
         [0.1306, 0.5276],
         [0.1061, 0.5523],
         [0.2446, 0.4007],
         [0.1670, 0.4770],
         [0.2485, 0.4313],
         [0.1227, 0.4909],
         [0.1240, 0.5668],
         [0.1461, 0.5113],
         [0.2315, 0.3788],
         [0.0494, 0.5590],
         [0.1107, 0.4799],
         [0.1121, 0.5735],
         [0.1007, 0.6318]])
    
    # 初始化 KMeans 模型并拟合数据
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    
    # 预测新数据点的所属簇
    new_points = np.array([[0.2567, 0.4326],[0.1956, 0.4280]])
    predictions = kmeans.predict(new_points)
    print(predictions)
