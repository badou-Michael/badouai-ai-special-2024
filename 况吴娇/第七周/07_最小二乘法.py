import pandas as pd

sales=pd.read_csv('train_data.csv',sep='\s*,\s*',engine='python')  #读取CSV
#engine 参数用于指定解析CSV文件时使用的引擎。pandas 提供了两个引擎选项：python 和 c。c 引擎是默认的，它是用C语言编写的，因此通常比 python 引擎更快。
#从名为 train_data.csv 的文件中读取数据。
# 使用正则表达式 \s*,\s* 作为字段的分隔符，这意味着字段可以由任意数量的空白字符包围的逗号分隔。
# 使用 python 引擎来解析文件，因为我们需要使用正则表达式作为分隔符。
X=sales['X'].values    #存csv的第一列
Y=sales['Y'].values    #存csv的第二列
#sales['Y']：这部分代码访问 DataFrame sales 中名为 'Y' 的列。结果是一个 Series 对象，包含了所有 'Y' 列的数据。

# data = {
#     'X': [1, 2, 3, 4],
#     'Y': [6, 5, 7, 10]
# }

print('X',X)
print('Y',Y)


#初始化赋值
s1 = 0
s2 = 0
s3 = 0
s4 = 0
n = 4       ####你需要根据的数据量进行修改 n为数据点的数量。

#循环累加
###通过循环遍历每个数据点，计算四个累加和，这些累加和将用于后续计算斜率和截距。
##易错点：确保循环次数n与数据点数量一致。
# s1=∑(Xi⋅Y i)
# s2=∑Xi
# s3=∑Yi
# s4=∑Xi^2
for i in range(n):
    s1 = s1 + X[i]*Y[i]     #X*Y，求和
    s2 = s2 + X[i]          #X的和
    s3 = s3 + Y[i]          #Y的和
    s4 = s4 + X[i]*X[i]     #X**2，求和


#使用最小二乘法公式计算斜率k和截距b。
##易错点：分母不能为零，否则会导致除以零的错误。需要确保数据点不完全相同，或者至少有一个数据点的X值不为零。
#计算斜率和截距
# k=( n(∑XiYi)−(∑Xi)(∑Yi) ）/ (n(∑Xi^2) - (∑Xi^2))
# b=(∑Yi−k(∑Xi) )/n

#k = (s2*s3-n*s1)/(s2*s2-s4*n)
k = (n*s1-s2*s3)/(s4*n-s2*s2)
b = (s3 - k*s2)/n
print('k斜率',k)
print('b截距',b)

print("Coeff: {} Intercept: {}".format(k, b))
#y=1.4x+3.5
