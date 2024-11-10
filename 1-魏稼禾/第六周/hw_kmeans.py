import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("lenna.png")
src = img.copy()

src = src.reshape(-1, 3)
src = np.float32(src)     # kmeans计算需要数据是float

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,
            10,0.1)
flags = cv2.KMEANS_PP_CENTERS

compactness, labels, centers = cv2.kmeans(src, 4, None, criteria, 10, flags)
dst4 = centers[labels.squeeze()]    # 使用np中的高级索引
dst4 = np.uint8(dst4.reshape(img.shape))    # 转换成512*512*3

compactness, labels, centers = cv2.kmeans(src, 8, None, criteria, 10, flags)
dst8 = centers[labels.squeeze()]
dst8 = np.uint8(dst8.reshape(img.shape))

compactness, labels, centers = cv2.kmeans(src, 16, None, criteria, 10, flags)
dst16 = centers[labels.squeeze()]
dst16 = np.uint8(dst16.reshape(img.shape))

imgs = [img, dst4, dst8, dst16]
titles = ["原图","4簇","8簇","16簇"]

# 正常显示标签
plt.rcParams["font.sans-serif"] = ["SimHei"]
for i in range(4):
    output = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
    print(output.shape)
    plt.subplot(2,2,i+1)
    plt.imshow(output)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()