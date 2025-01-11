# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

def kmeans_segmentation(image, num_clusters, criteria, flags):
    # 图像二维像素转换为一维
    data = image.reshape(-1, 3).astype(np.float32)
    
    # 执行K-means聚类
    compactness, labels, centers = cv2.kmeans(data, num_clusters, None, criteria, 10, flags)
    
    # 将聚类中心转换为整数类型
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    dst = res.reshape((image.shape))
    
    return dst

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    
    # 停止条件 (type, max_iter, epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    
    # 不同聚类数量
    k_values = [2, 4, 8, 16, 64]
    results = []
    
    for k in k_values:
        dst = kmeans_segmentation(img, k, criteria, flags)
        results.append(dst)
    
    # 图像转换为RGB显示
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = [cv2.cvtColor(dst, cv2.COLOR_BGR2RGB) for dst in results]

    # 显示图像
    titles = [u'original', u'K=2', u'K=4', u'K=8', u'K=16', u'K=64']
    images = [img] + results
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    
    plt.show()
