import math
import random
import numpy as np
import matplotlib.pyplot as plt

# K-Means算法计算距离
def Distance(src, dst):
    sum = 0
    for i in range(len(src)):
        sum += (src[i] - dst[i])**2
    return math.sqrt(sum)

# 函数：实现聚类算法
def Clustering(k, src, turn):
    clu = []
    for i in range(k):
        clu.append([])
        clu[i].append(src[random.randint(0, len(src)-1)])
    print("k类的中心值：", clu)
    for i in range(len(src)):
        flag = 999
        num = 0
        for j in range(k):
            dis = Distance(src[i], clu[j][0])
            if dis < flag:
                flag = dis
                num = j
        if clu[num][0] != src[i]:
            clu[num].append(src[i])
    # 计算k个簇中的每个均值点
    mean = []
    new = clu
    print("clu值：", new)
    round = 0
    while round != turn:
        for n in new:
            sum_x = 0
            sum_y = 0
            print("n:", n)
            for i in n:
                print("i:", i)
                sum_x += i[0]
                sum_y += i[1]
            print("sumx:", sum_x)
            print("sumy:", sum_y)
            print("len_n:", len(n))
            mean.append([sum_x / len(n), sum_y / len(n)])
        print("mean:", mean)
        new_clu = [[] for n in mean]
        for c in new:
            for d in c:
                max_dst = 999
                num = 0
                for j in range(k):
                    dst = Distance(mean[j], d)
                    if dst < max_dst:
                        max_dst = dst
                        tag = d
                        num = j
                new_clu[num].append(tag)
        new = new_clu;
        mean = []
        round = round + 1
        print("new:", new)
    return new

if __name__ == '__main__':
    k = 2
    turn = 20
    # src = [[2, 3], [5, 6], [2, 1], [1, 6], [2, 8], [1, 8], [2, 9]]
    src = [[0.0888, 0.5885],
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
    # 聚类前的图像
    x_old = [n[0] for n in src]
    y_old = [n[1] for n in src]
    print("x_old:", x_old)
    print("y_old:", y_old)
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.scatter(x_old, y_old, marker='s', color='b')
    plt.xlabel("x_old")
    plt.ylabel("y_old")
    clu = Clustering(k, src, turn)
    # 聚类后的图像
    color = ['r', 'g', 'b', 'y', 'k']
    marker = ['x', 's', 'd', 'v', 'p']
    plt.figure(1)
    plt.subplot(1, 2, 2)
    for i in range(len(clu)):
        x_new = []
        y_new = []
        for n in clu[i]:
            x_new.append(n[0])
        for n in clu[i]:
            y_new.append(n[1])
        plt.scatter(x_new, y_new, marker=marker[i], color=color[i])
    print("x_new:", x_new)
    print("y_new:", y_new)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

