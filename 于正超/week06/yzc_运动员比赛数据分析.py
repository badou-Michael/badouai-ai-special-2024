#coding = UTF-8
from sklearn.cluster import KMeans
#导入数据集，每行表示一位运动员数据，两列，分别表示球员每分钟助攻数、每分钟得分数
X = [[0.0888, 0.5885],
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
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]
    ]
print(X)
#KMeans 聚类
# clf = KMeans(n_clusters=3) 表示类簇数为3，聚成3类数据，clf即赋值为KMeans
# y_pred = clf.fit_predict(X) 载入数据集X，并且将聚类的结果赋值给y_pred

y_pred = KMeans(n_clusters=3).fit_predict(X)
print("聚类结果：",y_pred)

#可视化绘图
import matplotlib.pyplot as plt
x = [n[0] for n in X]
y = [n[1] for n in X]

plt.scatter(x,y,c=y_pred,marker='x')  #散点图，marker类型:o表示圆点,*表示星型,x表示点

plt.title("KMeans Data")
plt.xlabel("zhugong/min")
plt.ylabel("score/min")

#设置右上角图例
plt.legend(["A","BB","CCC"])
plt.show()


