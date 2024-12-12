from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
import matplotlib.pyplot as plt

'''
层次法(Hierarchical methods)先计算样本之间的距离。
每次将距离最近的点合并到同一个类。
然后，再计算类与类之间的距离，将距离最近的类合并为一个大类。
不停的合并，直到合成了一个类。

scipy.cluster.hierarchy
linkage(y, method='single', metric='euclidean', optimal_ordering=False)

X
	输入数据，二维数组或距离矩阵
	如果 X 是一个二维数组，形状为 (n_samples, n_features)，表示有 n_samples 个样本，每个样本有 n_features 个特征。
	如果 X 是一个距离矩阵，形状为 (n_samples, n_samples)，表示每对样本之间的距离。在这种情况下，metric 参数将被忽略。
method
	指定了如何计算两个簇之间的距离，并决定如何合并簇。
	single
		单链法
			
			使用最短距离法，即两个簇之间的距离是它们最近的两个点之间的距离。这种方法容易形成“链状”簇。
	complete
		全链法
			
			使用最长距离法，即两个簇之间的距离是它们最远的两个点之间的距离。这种方法倾向于形成紧凑的簇。
	average
		平均链法
			
			使用平均距离法，即两个簇之间的距离是所有点对之间的平均距离。这种方法通常介于单链法和全链法之间。
	weighted
		加权平均法
			
			类似于平均链法，但每次合并时，新簇的距离是基于之前合并的簇的平均距离，而不是所有点对的平均距离。这种方法可以减少不平衡簇的影响。
	centroid
		质心法
			
			使用簇的质心之间的欧几里得距离。这种方法假设簇是球形的，可能会导致不稳定的聚类结果。
			其中Cu和Cv分别是簇u和v的质心。
	median
		中位数法
			
			类似于质心法，但每次合并时，新簇的质心是前两个簇质心的中点。这种方法也可以假设簇是球形的。
	ward
		沃德最小方差法
			
			最小化簇内方差的增加。这种方法适用于欧几里得距离，并且倾向于生成紧凑且大小均匀的簇。
metric
	指定了如何计算样本之间的距离。
	euclidean
		
		欧几里得距离（默认值）
	cityblock
    manhattan
		
		曼哈顿距离（L1 距离）
	cosine
		
		余弦相似度（1 - 余弦相似度）
	correlation
		
		相关系数距离（1 - 相关系数）
	hamming
		
		汉明距离（适用于二进制数据）
	jaccard
		
		Jaccard 距离（适用于二进制数据）
	precomputed
		如果 X 是一个预计算的距离矩阵，则使用这个选项。此时 metric 参数将被忽略。
optimal_ordering
	是否优化树状图的顺序
	布尔值，默认为False
	如果设置为 True，linkage 会尝试优化树状图的顺序，使得相邻的叶子节点在树状图中尽可能接近。这可以提高树状图的可读性，但会增加计算时间。
返回一个形状为 (n-1, 4) 的链接矩阵 Z，其中 n 是原始数据点的数量。每一行表示一次合并操作，包含以下信息：
第一列
	合并的第一个簇的索引。如果是原始数据点，则索引范围为 [0, n-1]；如果是之前合并形成的簇，则索引范围为 [n, 2n-2]。
第二列
	合并的第二个簇的索引。规则同上。
第三列
	这两个簇之间的距离。
第四列
	新簇中的样本数量。

引入
	scipy.cluster.hierarchy
函数签名
	clusters = fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None)
参数说明
	Z
		输入数据，链接矩阵（linkage matrix）
		二维数组
	t
		阈值参数
			浮点数或整数
		t 的具体含义取决于 criterion 参数的选择。它可以是一个距离阈值、簇的数量或其他标准。
	criterion
		切分标准
			默认值为 'inconsistent'
		决定如何使用 t 参数来切分层次聚类的关键参数。
		inconsistent
			基于不一致性统计量（inconsistency statistics）来确定簇的边界。depth 参数指定了计算不一致性统计量时考虑的层级深度。
		distance
			根据距离阈值来切分层次聚类。所有距离小于或等于 t 的簇将被合并为一个簇。这种方法适用于你有一个明确的距离阈值标准的情况。
		maxclust
			根据簇的数量来切分层次聚类。t 参数指定你希望得到的簇的数量。fcluster 会尝试将数据切分为 t 个簇。
		maxclust_monocrit
			根据簇的数量和单一准则（如簇内的最大距离）来切分层次聚类。monocrit 参数需要提供一个数组，指示每个簇的单一准则值。
		monocrit
			基于单一准则（如簇内的最大距离）来切分层次聚类。monocrit 参数需要提供一个数组，指示每个簇的单一准则值。
	depth
		不一致性统计量的深度
			整数
			默认值为 2
		仅在 criterion='inconsistent' 时有效。
		它指定了计算不一致性统计量时考虑的层级深度。较大的 depth 值会使不一致性统计量更加平滑，但也会增加计算复杂度。
	R
		不一致性统计量矩阵
			二维数组
			默认值为 None
		仅在 criterion='inconsistent' 时有效。
		这是一个可选的参数，提供了预计算的不一致性统计量矩阵。如果你已经计算了这个矩阵，可以传递给 fcluster 以加快计算速度。
	monocrit
		单一准则值
			一维数组
			默认值为 None
		仅在 criterion='monocrit' 或 'maxclust_monocrit' 时有效。
		它是一个数组，指示每个簇的单一准则值（如簇内的最大距离）。fcluster 会根据这些值来切分层次聚类。

返回一个长度为 n 的整数数组，表示每个数据点所属的簇编号。簇编号从 1 开始，不同簇之间没有重复编号。如果两个数据点属于同一个簇，它们的簇编号相同；否则，簇编号不同。

'''

X = np.array([[np.random.randint(0, 10), np.random.randint(0, 10)] for _ in range(80)])
# print(X)

Z = linkage(X, method='ward')
f = fcluster(Z, 4, criterion='distance')
dn = dendrogram(Z)
print(Z)
plt.show()






