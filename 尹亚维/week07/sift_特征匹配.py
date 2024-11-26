"""
SIFT算法是一种用于图像处理和计算机视觉中提取图像特征的算法，
具有尺度不变性和旋转不变性，广泛应用于图像匹配、目标识别等领域
在较新的 OpenCV 版本中，cv2.SIFT() 可能已经被 cv2.SIFT_create() 替代，
因为 SIFT 算法涉及专利问题，某些版本的 OpenCV 可能不直接支持 cv2.SIFT()
SURF 是 SIFT 的一种快速替代算法，旨在提高特征检测的速度
"""
import cv2
import numpy as np


def drawMatchesKnn_cv2(img1_gray, img2_gray, kp1, kp2, goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]
    # 创建一个空白的三通道图像，用于显示匹配结果
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    # 将第一幅图像复制到可视化图像的左半部分
    vis[:h1, :w1] = img1_gray
    # 将第二幅图像复制到可视化图像的右半部分
    vis[:h2, w1:w1 + w2] = img2_gray

    # 提取匹配点在第一幅图像中的索引
    p1 = [kpp.queryIdx for kpp in goodMatch]
    # 提取匹配点在第二幅图像中的索引
    p2 = [kpp.trainIdx for kpp in goodMatch]

    # 获取第一幅图像中匹配点的坐标
    post1 = np.int32([kp1[pp].pt for pp in p1])
    # 获取第二幅图像中匹配点的坐标，并将其向右平移 w1 个像素
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)
    for (x1, y1), (x2, y2) in zip(post1, post2):
        # 使用 cv2.line 函数在可视化图像上绘制从第一幅图像到第二幅图像的匹配线，颜色为红色
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    # 创建一个可调整大小的窗口
    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    # 显示包含匹配线的可视化图像
    cv2.imshow("match", vis)


img1_gray = cv2.imread('iphone1.png')
img2_gray = cv2.imread('iphone2.png')
sift = cv2.xfeatures2d.SIFT_create()
# cv2.xfeatures2d.SIFT_create(nfeatures=100, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
# 参数如下
# nfeatures：要检测的关键点数量，默认值为 0（表示检测所有关键点）。
# nOctaveLayers：每个八度层的层数，默认值为 3。
# contrastThreshold：对比度阈值，用于过滤低对比度的关键点，默认值为 0.04。
# edgeThreshold：边缘响应阈值，默认值为 10。
# sigma：高斯滤波器的标准差，默认值为 1.6。
# sift = cv2.SIFT()
# sift = cv2.SURF()

kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)
# 执行关键点检测和描述符计算,得到kp1 是检测到的关键点列表，des1 是对应的描述符矩阵

# BFmatcher with default parms
bf = cv2.BFMatcher(cv2.NORM_L2)
# opencv中knnMatch是一种蛮力匹配
# 将待匹配图片的特征与目标图片中的全部特征全量遍历，找出相似度最高的前k个。
matches = bf.knnMatch(des1, des2, k=2)
goodMatch = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        goodMatch.append(m)

drawMatchesKnn_cv2(img1_gray, img2_gray, kp1, kp2, goodMatch[:20])

cv2.waitKey(0)
cv2.destroyAllWindows()
