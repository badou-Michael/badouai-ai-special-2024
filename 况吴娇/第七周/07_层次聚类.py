###cluster.py
#导入相应的包
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。

'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''


###SciPy是一个开源的Python算法库和数学工具包。它建立在NumPy的基础上，用于进行科学计算和技术计算。
# SciPy提供了许多用于优化、线性代数、积分、插值、特殊函数、快速傅里叶变换、信号处理和图像处理等模块。

#Matplotlib是一个用于创建静态、交互式和动画可视化的Python库。
# 它提供了一个类似于MATLAB的绘图框架，使得绘图变得简单直观。


# scipy.cluster.hierarchy模块中导入了三个函数：
# dendrogram：用于绘制树状图（dendrogram），这是一种用于展示层次聚类结果的图形表示方法。
# linkage：用于执行层次聚类算法。这个函数会根据提供的距离矩阵或数据点计算出聚类树。
# fcluster：用于从聚类树中提取聚类。这个函数可以根据给定的阈值或聚类数来切割树状图，从而得到最终的聚类结果。

X=[[1,2],[3,2],[4,4],[1,2],[1,3]]
print('X输入',X)
Z=linkage(X,'ward') ##linkage(X,'ward')（Linkage method），即如何计算聚类之间的距离。'ward'：Ward's method最小化聚类内的方差。
print('linkage最小化聚类内的方差',Z)

f =fcluster(Z,4,'distance')
##fcluster 函数用于从聚类树 Z 中提取聚类。这里我们指定要提取4个聚类（4），并且使用距离作为聚类切割的依据（'distance'）。
#t:4 ;criterion:'distance' 表示
fig =plt.figure(figsize=(5,3))  ##figsize=(5,3) 表示创建的图形宽度为5英寸，高度为3英寸。这个尺寸是整个图形窗口的大小，包括边缘、标题、轴标签等。
#plt.figure() 函数创建一个新的图形窗口。如果不指定 figsize，默认会使用一个较小的尺寸，这可能会导致图形中的元素（如轴、标签、图例等）显得拥挤，特别是当图形复杂或者需要显示大量数据时。
dn = dendrogram(Z)

plt.show()



#dendrogram 是 scipy.cluster.hierarchy 模块中的一个函数，用于绘制由 linkage 函数生成的聚类树状图（dendrogram）。
# 这个树状图是一种树形结构的图表，用于展示层次聚类的结果，其中每个分支代表一个聚类，分支的合并点代表聚类之间的合并。
#dendrogram(Z) 函数生成了树状图，并且 dn 变量保存了这个图形的信息。然后，plt.show() 函数被调用来在屏幕上显示这个图形。如果你不调用 plt.show()，图形将不会显示出来。