# MTCNN
'''
MTCNN是一种兼顾效率与准确性的计算机视觉人脸检测和矫正模型，由P-Net、R-Net、O-Net三个级联网络组成。
通过图像金字塔处理，MTCNN能检测不同大小的人脸，并进行人脸对齐。每个网络层分别进行人脸分类、边界框回归
和特征点定位，使用在线困难样本挖掘策略提升训练精度。
利用滑动窗口的方法，在图片当中将小框框每次向右移动一定的长度，然后获得下一个框框，当框框移动到第一行的
最后的时候，则第一行取完了 之后向下平移，此时当做第二行。简单总结就是将框框按照一定的长度平移，遍历整张
图片，从而找到人脸的位置。

MTCNN将原始图像缩放到不同尺度，形成图像金字塔。

MTCNN由三个级联的网络组成，分别是P-Net, R-Net, O-Net图片经过预处理，先经过P-Net网络，将结果给R-Net网络，
R-Net网络的输出再传给O-Net网络，O-Net最后得到输出结果。

MTCNN的三层结构：
阶段一：先使用全卷积网络，即P-Net，来获取到获选的人脸框和其对应的向量。随后根据回归框向量对候选框进行校正。之后使用
非极大抑制（NMS）来去除高度重合的候选框。
阶段二：P-Net得到的回归框都送入到R-Net中，随后拒绝大量错误框，再对回归框做校正，并使用NMS去除重合框。
阶段三：与阶段二类似，但是这里会额外进行人脸特征点（5个）的检测。

模型结构
用三个小模型级联：
小模型P-Net快速排除图片中不含人脸的部分。
中模型R-Net进一步排除不含人脸的部分。
大模型O-Net敲定人脸的位置，重叠的框用NMS除去并且标记人脸的左右眼、鼻、两嘴角 共五个位置。
P-Net
判断这个输入的图像中是否有人脸，并且给出人脸框关键点的位置。
R-Net
以P-Net预测得到的回归框信息作为输入，先对原始图片进行切片，随后resize到固定尺寸shape=(24x24x3)。
输出和P-Net输出一样，人脸分类，边界框回归，人脸特征点定位。
O-Net
以R-Net预测的bbox信息作为输入，对原始图片进行切片，并resize到固定尺寸shape=(48x48x3)。
输出与R-Net输出一样。

P-Net 、R-Net、O-Net，网络输入的图片越来越大，卷积层的通道数越来越多，内部的层数也越来越多，P-Net运行速度最快，O-Net运行很慢。

之所以要使用三个网络，是因为如果一开始直接对图中的每个区域使用O-Net，速度会很慢，实际上P-Net先做一遍过滤将过滤后的结果再交给R-Net
进行过滤，最后将过滤后的结果交给效果最好的，但速度较慢的O-Net进行判别。这样在每一步都提前减少了需要判别的数量，有效降低了处理时间。

从P-Net到R-Net再到O-Net，网络的输入图像尺寸越来越大，结构越来越深，提取的特征也越具有表现能力。通过不同的损失函数结构设计，每个网
络在应用端实现不同的功能。


'''

import cv2
import numpy as np
from mtcnn import mtcnn #调用mtcnn函数

img = cv2.imread('mtcnn_img/timg.jpg')

model = mtcnn()
threshold = [0.5,0.7,0.8]  # 三段网络的置信度阈值不同
rectangles = model.detectFace(img, threshold)
draw = img.copy()

for rectangle in rectangles:
    if rectangle is not None:
        W = -int(rectangle[0]) + int(rectangle[2])
        H = -int(rectangle[1]) + int(rectangle[3])
        paddingH = 0.01 * W
        paddingW = 0.02 * H
        crop_img = img[int(rectangle[1]+paddingH):int(rectangle[3]-paddingH), int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]
        if crop_img is None:
            continue
        if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
            continue
        cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 1)

        for i in range(5, 15, 2):
            cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))

cv2.imwrite("mtcnn_img/out.jpg",draw)

cv2.imshow("test", draw)
c = cv2.waitKey(0)