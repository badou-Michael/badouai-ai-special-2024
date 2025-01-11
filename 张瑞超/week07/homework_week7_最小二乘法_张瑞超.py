import pandas as pd

# sep='\s*,\s*' 正则表达式，表示用逗号作为分隔符，并且允许逗号前后有空格或空白字符
# engine：指定解析引擎。engine='python' 表示使用 Python 解析器，支持更复杂的分隔符如正则表达式。
sales=pd.read_csv('train_data.csv', sep='\s*,\s*', engine='python')
X=sales['X'].values    #存csv的第一列
Y=sales['Y'].values    #存csv的第二列

#初始化赋值
s1 = 0
s2 = 0
s3 = 0
s4 = 0
# 数据量n从数据的长度自动获取
n = len(X)

#循环累加
for i in range(n):
    s1 = s1 + X[i]*Y[i]     #X*Y，求和
    s2 = s2 + X[i]          #X的和
    s3 = s3 + Y[i]          #Y的和
    s4 = s4 + X[i]*X[i]     #X**2，求和

#计算斜率和截距
k = (s2*s3-n*s1)/(s2*s2-s4*n)
b = (s3 - k*s2)/n
print("Coeff: {} Intercept: {}".format(k, b))