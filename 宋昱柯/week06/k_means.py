import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


pic_path = "practice\cv\week06\lenna.png"
img = cv.imread(pic_path, 0)

rows, cols = img.shape
data = img.reshape(rows * cols).astype(np.float32)
# 停止条件
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,10,1.0)
# 标签
flag = cv.KMEANS_RANDOM_CENTERS
# k-means聚类
compactness, labels, centers = cv.kmeans(data, 4, None, criteria, 10, flag)
print(f"compatness = {compactness}")
print(f"labels = {labels}")
print(f"centers = {centers}")

dst = labels.reshape((rows, cols))
# dst = dst * 255 // (4 - 1)
print(dst)
plt.rcParams["font.sans-serif"] = ["SimHei"]

# 显示图像
titles = ["原始图像", "聚类图像"]
images = [img, dst]
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], "gray"),
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
