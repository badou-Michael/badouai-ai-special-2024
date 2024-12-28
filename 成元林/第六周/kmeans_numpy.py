import numpy as np
import matplotlib.pyplot as plt

def do_kmeans(data,centers,sumDis):
    class_data,sumdis_new = classfy(data,centers)
    new_centers = updateCenters(class_data)
    if sumDis == sumdis_new:
        for i in range(len(class_data)):
            per_class = class_data[i]
            for j in per_class:
                plt.scatter(j[0], j[1], c=color_list[i])
        plt.show()
        return
    # for i in range(new_centers.shape[0]):
    #     center = new_centers[i]
    #     print(center)
    #     plt.scatter(center[0],center[1],marker="X",c=color_list[i])
    # for i in range(len(class_data)):
    #     per_class = class_data[i]
    #     for j in per_class:
    #         plt.scatter(j[0],j[1],c=color_list[i])
    # plt.show()
    do_kmeans(data,new_centers,sumdis_new)

def classfy(data, centers):
    class_data = [[] for i in range(centers.shape[0])]
    sumdis = 0
    for i in range(len(data)):
        perdata = data[i]
        diffData1 = perdata - centers[0]
        diffData2 = perdata - centers[1]
        diffData3 = perdata - centers[2]
        distance1 = np.sqrt(diffData1[0] ** 2 + diffData1[1] ** 2)
        distance2 = np.sqrt(diffData2[0] ** 2 + diffData2[1] ** 2)
        distance3 = np.sqrt(diffData3[0] ** 2 + diffData3[1] ** 2)
        sumdis += distance1 + distance2 + distance3
        distance_data = [distance1, distance2, distance3]
        min_index = np.argsort(distance_data)
        class_data[min_index[0]].append(perdata)
    return class_data, sumdis


def updateCenters(class_data):
    centers = []
    for j in range(len(class_data)):
        per_cldata = class_data[j]
        per_cldata = np.array(per_cldata)
        avg_data = np.sum(per_cldata, axis=0) / len(per_cldata)
        centers.append(avg_data)
    return np.array(centers)

if __name__ == '__main__':
    color_list = ["yellow","red","pink","black","blue"]
    data = [[0.0888, 0.5885],
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
    data_ln = len(data)
    centers = []
    k = 3
    for i in range(k):
        r_data_index = np.random.randint(0, data_ln)
        centers.append(data[r_data_index])
    centers = np.array(centers)
    do_kmeans(data,centers,0)


