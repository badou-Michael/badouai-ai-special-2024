import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png")       # cv2获得的是BGR！
# 生成img的灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 对gray做直方图均衡化,dst是均衡化后[各灰度级会分布得“更均匀”]的图像
dst = cv2.equalizeHist(gray)
# 查看dst的灰度直方图——https://blog.csdn.net/star_sky_sc/article/details/122371392
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])       # [0]表示灰度图；[0.256]——前包含后不含→256不会被取到
# 输出展示生成的直方图hist
plt.figure()
plt.hist(dst.ravel(), 256)      # 直方图hist函数只支持一维数组；img.ravel()可以把多维数组转化成一维数组；256 表示横坐标的最大值为256，有256条柱
plt.show()
# 输出展示均衡化后的图像
# np.hstack()——numpy库中的一个数组堆叠函数→进行图像拼接；np.hstack()-水平拼接、np.vstack()-垂直拼接
# np.hstack() 按水平方向（列顺序）拼接 2个或多个图像，图像的高度（数组的行）必须相同
# np.vstack() 按垂直方向（行顺序）拼接 2个或多个图像，图像的宽度（数组的列）必须相同
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)



# 彩色图像直方图均衡化
# img = cv2.imread("lenna.png")

# 彩色图像均衡化→先分解通道，再对每一个通道做直方图均衡化
(b, g, r) = cv2.split(img)      # cv2.split()→可以拆分图像的通道
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))

cv2.imshow("src", img)
cv2.imshow("dst_rgb", result)
cv2.waitKey(0)


