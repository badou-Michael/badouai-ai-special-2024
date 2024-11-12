# -*- coding: utf-8 -*-
# time: 2024/11/5 14:54
# file: SIFT_特征匹配.py
# author: flame
import cv2
import numpy as np

"""
读取两张图像，检测并匹配它们之间的特征点，并在一张合并的图像上绘制这些匹配的特征点。
1. 读取两张图像。
2. 使用SIFT算法检测和计算图像的特征点和描述符。
3. 使用BFMatcher进行特征点匹配。
4. 筛选出高质量的匹配点。
5. 绘制匹配的特征点。
"""

""" 定义一个函数，用于绘制两幅图像中的特征点匹配情况 """
def drawMatching(img1, img2, kp1, kp2, matches):
    """
    在两幅图像中绘制匹配的特征点。

    参数:
    img1 -- 第一幅图像
    img2 -- 第二幅图像
    kp1 -- 第一幅图像中的关键点
    kp2 -- 第二幅图像中的关键点
    matches -- 匹配的关键点列表
    """
    """ 获取两幅图像的高度和宽度，用于后续创建画布。 """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    """ 创建一个足够大的画布来放置两幅图像，画布大小为 (max(h1, h2), w1 + w2, 3)，类型为 uint8。 """
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    """ 将第一幅图像复制到画布的左半部分。 """
    vis[:h1, :w1] = img1
    """ 将第二幅图像复制到画布的右半部分。 """
    vis[:h2, w1:w1 + w2] = img2

    """ 获取匹配关键点的索引，queryIdx 是查询图像中的关键点索引，trainIdx 是训练图像中的关键点索引。 """
    p1 = [kpp.queryIdx for kpp in matches]
    p2 = [kpp.trainIdx for kpp in matches]

    """ 获取匹配关键点的坐标，并将第二幅图像的坐标进行偏移，使其在画布的右半部分显示。 """
    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    """ 在匹配的特征点之间绘制红色连线，使用 cv2.line 函数。 """
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    """ 创建一个名为 'match' 的窗口，并设置窗口大小为自动调整。 """
    cv2.namedWindow("match", cv2.WINDOW_AUTOSIZE)
    """ 在窗口中显示绘制了匹配特征点的图像。 """
    cv2.imshow("match", vis)

""" 读取第一张图像，使用 cv2.imread 函数，返回一个 NumPy 数组。 """
img1 = cv2.imread("iphone1.png")

""" 读取第二张图像，使用 cv2.imread 函数，返回一个 NumPy 数组。 """
img2 = cv2.imread("iphone2.png")

""" 创建 SIFT 对象，用于检测和计算图像的特征点和描述符。 """
sift = cv2.xfeatures2d.SIFT_create()

""" 检测第一张图像的特征点和计算描述符，返回关键点列表和描述符数组。 """
kp1, des1 = sift.detectAndCompute(img1, None)

""" 检测第二张图像的特征点和计算描述符，返回关键点列表和描述符数组。 """
kp2, des2 = sift.detectAndCompute(img2, None)

""" 创建 BFMatcher 对象，用于匹配特征点，使用 L2 距离度量。 """
bf = cv2.BFMatcher(cv2.NORM_L2)

""" 进行特征点匹配，返回每个特征点的两个最近邻，k=2 表示返回两个最近邻。 """
matches = bf.knnMatch(des1, des2, k=2)

""" 筛选出高质量的匹配点，即第一个最近邻的距离小于第二个最近邻距离的 0.5 倍。 """
goodMatch = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        goodMatch.append(m)

""" 调用 drawMatching 函数，绘制前 20 个高质量的匹配点。 """
drawMatching(img1, img2, kp1, kp2, goodMatch[:20])

""" 等待用户按键，然后关闭所有窗口，使用 cv2.waitKey 和 cv2.destroyAllWindows 函数。 """
cv2.waitKey(0)
cv2.destroyAllWindows()
