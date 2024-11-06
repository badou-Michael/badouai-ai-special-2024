import cv2
import matplotlib.pyplot as plt

# 灰度图像直方图
# 获取灰度图像
img = cv2.imread('lenna.png', 1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("image gray", img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 灰度图像的直方图，方法一
# plt.figure()
# plt.hist(img_gray.ravel(), 256)
# plt.show()

# 灰度图像的直方图, 方法二
# [img_gray]：表示输入的灰度图像。img_gray 是一个灰度图像，通常通过 cv2.cvtColor 转换得到。
# [0]：表示计算直方图的通道索引。对于灰度图像，通道索引为 0。
# None：表示没有掩膜（mask）。如果需要对特定区域进行直方图计算，可以传入一个掩膜图像。
# [256]：表示直方图的 bins 数量。这里设置为 256，意味着直方图将被划分为 256 个区间。
# [0, 256]：表示直方图的范围。灰度图像的像素值范围从 0 到 255，因此这里设置为 [0, 256]。
# hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
# plt.figure()  # 新建一个图像
# plt.title("Grayscale Histogram")
# plt.xlabel("Bins")  # X轴标签
# plt.ylabel("# of Pixels")  # Y轴标签
# plt.plot(hist)
# plt.xlim([0, 256])  # 设置x坐标轴范围
# plt.show()

# 彩色图像直方图
# cv2.imshow("Original", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# channels = cv2.split(img)
# colors = ('b', 'g', 'r')
# plt.figure()
# plt.title("Flattened Color Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# for (channel,color) in zip(channels, colors):
#     hist = cv2.calcHist([channel], [0], None, [256], [0,256])
#     plt.plot(hist, color = color)
#     plt.xlim([0, 256])
# plt.show()
