
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt
import random

X = []
# 获取用户输入
num_tuples = int(input("请输入你想随机生成的元组的数量："))
for _ in range(num_tuples):
    # 生成一个随机元组，这里假设元组包含两个随机整数
    random_tuple = (random.randint(1, 100), random.randint(1, 100))
    X.append(random_tuple)
Z = linkage(X, 'ward')
f = fcluster(Z,4,'distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
plt.show()
