# -*- coding: utf-8 -*-
# author: 王博然
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    data = np.array([[1,2],[3,2],[4,4],[1,2],[1,3]])
    z = linkage(data, 'ward')
    plt.figure()
    dendrogram(z)
    plt.title('Dendrogram')
    plt.show()