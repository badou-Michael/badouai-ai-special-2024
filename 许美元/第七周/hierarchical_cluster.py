

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

data_list = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]

result_list = linkage(data_list, 'ward')
'''
linkage 函数是 SciPy 库中用于层次聚类的函数之一，
它执行的是凝聚式层次聚类（agglomerative hierarchical clustering）。
凝聚式层次聚类是一种自底向上的聚类方法，它从每个数据点作为单独的聚类开始，
然后逐步合并聚类，直到所有数据点都合并成一个大的聚类，或者满足某个停止条件。

【参数解释】
1.data_list：待聚类的数据，通常是一个二维数组，其中每一行代表一个观测值，每一列代表一个特征。

2.method：指定聚类时使用的链接方法。常用的链接方法包括：
    'single'：最近邻方法，也称为最小方法，选择两个聚类中最近的两个点作为聚类之间的距离。
    'complete'：最远邻方法，也称为最大方法，选择两个聚类中最远的两个点作为聚类之间的距离。
    'average'：平均链接方法，计算两个聚类中所有点对之间的平均距离。
    'ward'：Ward 方法，最小化聚类内的方差。
    
3.metric：指定计算数据点之间距离的度量。常用的度量包括 'euclidean'（欧几里得距离）、'cityblock'（曼哈顿距离）等。

【返回值】
linkage 函数返回一个数组 result_list，其中每一行代表一次聚类合并事件。对于每次合并，result_list 数组的一行包含以下四个值：
    result_list[i, 0] 和 result_list[i, 1]：被合并的两个聚类的索引。
    result_list[i, 2]：合并后的新聚类与原始聚类之间的距离（根据指定的方法计算）。
    result_list[i, 3]：合并后的新聚类中的观测值数量。
'''
print(result_list)

f_result = fcluster(result_list, 4, 'distance')
'''
fcluster 函数是 SciPy 库中用于从层次聚类结果中形成平面（flat）聚类的函数。
以下是对该函数的详细解释：

函数定义：
scipy.cluster.hierarchy.fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None) 是 fcluster 函数的原型，它根据给定的链接矩阵 Z 和特定的标准 criterion 来形成平面聚类。

【参数】
    1.Z：由 linkage 函数返回的链接矩阵，该矩阵编码了层次聚类的结果。
    2.t：一个标量值，用于确定形成平面聚类的阈值。
    3.criterion：用于形成平面聚类的标准，可以是以下几种之一：
        'inconsistent'：如果一个聚类节点及其所有后代的不一致值小于或等于 t，则其所有叶子后代属于同一个平面聚类。如果没有非单例聚类满足此条件，则每个节点被分配到自己的聚类中（默认值）。
        'distance'：形成平面聚类，使得原始观测值在每个平面聚类中的共生距离不超过 t。
        'maxclust'：找到一个最小阈值 r，使得同一平面聚类中任意两个原始观测值之间的共生距离不超过 r，并且形成的平面聚类数量不超过 t。
        'monocrit'：当 monocrit[j] <= t 时，从具有索引 i 的聚类节点 c 形成平面聚类。
        'maxclust_monocrit'：当所有包括 c 在内及以下的聚类索引 i 的 monocrit[i] <= r 时，从非单例聚类节点 c 形成平面聚类。r 被最小化，使得不超过 t 个平面聚类被形成。
    4.depth：执行不一致性计算的最大深度，对于其他标准没有意义，默认值为 2。
    5.R：用于 'inconsistent' 标准的不一致性矩阵。如果未提供，则会被计算。
    6.monocrit：长度为 n-1 的数组，monocrit[i] 是用于对非单例 i 进行阈值处理的统计数据。monocrit 向量必须是单调的。
    
【返回值】
返回一个数组，包含每个数据点的聚类标签。
具体来说，结果 [1 2 2 1 1] 意味着：
    第一个数据点被分配到聚类 1。
    第二个数据点被分配到聚类 2。
    第三个数据点被分配到聚类 2。
    第四个数据点被分配到聚类 1。
    第五个数据点被分配到聚类 1。
'''
print(f_result)

fig = plt.figure(figsize=(5, 3))

dn = dendrogram(result_list)
'''
这行代码使用 dendrogram 函数根据聚类结果 Z 绘制树状图，并将结果存储在变量 dn 中。
树状图是一种可视化层次聚类结果的图形，显示了数据点是如何被逐步合并成聚类的。
'''
plt.show()