from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

X = [[1,2],[5,2],[4,5],[2,5],[1,6]]
Z = linkage(X,'ward')
f = fcluster(Z, 4,'distance')
print(Z)
print(f)
#创建了一个大小为 5英寸宽和5 英寸高的图形窗口
fig = plt.figure(figsize = (5,5))#
#绘制由 linkage 函数生成的层次聚类结果 Z 的树状图
dn = dendrogram(Z)
'''
Ward 方法的核心思想是最小化每个聚类内部的方差，以此来衡量聚类的效果。
当使用 Ward 方法进行层次聚类时，算法会尝试找到那些合并后增加的方差最小的两个聚类，并将它们合并。
具体来说，Ward 方法在每一步都会计算合并任意两个聚类后增加的方差（称为“增益”），
然后选择增益最小的那对聚类进行合并。这种方法试图找到一种聚类方式，
使得每个聚类内部的数据点尽可能相似（即方差最小），而不同聚类之间的差异尽可能大。
'''
plt.show()
'''
scipy`库中的`cluster.hierarchy`模块。
层次聚类是一种常用的聚类方法，它不需要预先指定聚类的数量，而是生成一个聚类树（称为树状图或dendrogram），
然后可以根据需要切割树状图来确定聚类的数量。
这些函数通常一起使用，首先使用`linkage`函数对数据进行层次聚类，
然后使用`dendrogram`函数绘制树状图以可视化聚类结果，
最后使用`fcluster`函数根据树状图的阈值来确定最终的聚类。

1. `linkage(X, method='single', metric='euclidean', optimal_ordering=False)`：
   - `X`：待聚类的数据，通常是一个二维数组，其中每一行代表一个观测值，每一列代表一个特征。
   - `method`：链接方法，指定如何合并聚类。常用的方法包括'single'（最近邻）、'complete'（最远邻）、'average'（平均距离）等。
   - `metric`：距离度量，指定计算数据点之间距离的方法。常用的度量包括'euclidean'（欧几里得距离）、'cityblock'（曼哈顿距离）等。
   - `optimal_ordering`：布尔值，指定是否在树状图中对数据点进行重新排序以优化树状图的显示。

2. `dendrogram(Z, truncate_mode='lastp', p=30, show_leaf_counts=True, leaf_rotation=90., leaf_font_size=12., 
show_contracted=True, link_color_func=None, ax=None, above_threshold_color='b', orientation='top', labels=None,
 color_threshold=None, max_d=0., **kwargs)`：
   - `Z`：由`linkage`函数返回的聚类结果。
   - `truncate_mode`：树状图的截断模式。
   - `p`：显示树状图时，显示前`p`个聚类。
   - `show_leaf_counts`：是否显示叶节点的计数。
   - `leaf_rotation`：叶节点标签的旋转角度。
   - `leaf_font_size`：叶节点标签的字体大小。
   - `show_contracted`：是否显示收缩的聚类。
   - `link_color_func`：自定义链接颜色的函数。
   - `ax`：指定绘制树状图的轴。
   - `above_threshold_color`：阈值以上链接的颜色。
   - `orientation`：树状图的方向。
   - `labels`：叶节点的标签。
   - `color_threshold`：颜色变化的阈值。
   - `max_d`：树状图显示的最大距离。

3. `fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None)`：
   - `Z`：由`linkage`函数返回的聚类结果。
   - `t`：切割树状图的阈值。
   - `criterion`：形成聚类的标准。
   - `depth`：在树状图中的深度。
   - `R`：一个数组，用于指定聚类的成员。
   - `monocrit`：一个函数，用于单参数的聚类。


'''