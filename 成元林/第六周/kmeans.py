import cv2
import numpy as np


def k_means_rgb(srcimg, k):
    tarimg = srcimg.copy()
    data = tarimg.reshape(-1, 3)
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 1.0)
    """
    compactness:紧密度，即每个点到其相应簇中心的距离的平方和。这个值越小，表示聚类效果越好‌
    labels:每个数据点的最终分类标签数组。这个数组指示了每个数据点属于哪个簇‌
    centers:聚类中心组成的数组。这些中心点代表了每个簇的平均位置‌
    """
    compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    print("centers:\n", centers)
    print("labels:\n", labels)
    print("labels.flatten():\n", labels.flatten())
    res = centers[labels.flatten()]
    print(res)
    dst = res.reshape(tarimg.shape)
    return dst


if __name__ == '__main__':
    srcimg = cv2.imread("lenna.png")
    tarimg = k_means_rgb(srcimg, 8)
    cv2.imshow("srcimg", srcimg)
    cv2.imshow("tarimg", tarimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
