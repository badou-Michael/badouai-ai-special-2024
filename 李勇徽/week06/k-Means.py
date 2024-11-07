import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
'''

def K_means(data, k, img_shape):
    # :params: k - 设置聚类类簇数
    # 进行K-means聚类
    # 设置criteria(type, max_iter, epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)
    # 图像转回uint8二维类型
    centers = np.uint8(centers)
    # 以聚类后得到的标签labels为索引取centers的值
    res = centers[labels.flatten()]
    # 重新reshape为图像的尺寸
    dst = res.reshape((img_shape))
    # 图像转换为RGB显示
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    title = 'K_means K=' + str(k)
    return dst, title

def BGR2RGB(img_BGR):
    return cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    # 读取原图像
    img = cv2.imread('lenna.png')
    titles = ['orginal image']
    images = [BGR2RGB(img)]
    # 二维图像转一维
    arr = img.reshape((-1, 3))
    data = np.float32(arr)

    # 聚类
    for i in range(1, 6):
        print('k: ', 2**i)
        k_img, title = K_means(data, 2 ** i, img.shape)
        images.append(k_img)
        titles.append(title)

    # 设置图像字体
    plt.rcParams['font.sans-serif']=['Georgia']
    #显示图像
    for i in range(6):  
       plt.subplot(2,3,i+1) 
       plt.imshow(images[i]) 
       plt.title(titles[i])  
       plt.axis('off')
    plt.show()