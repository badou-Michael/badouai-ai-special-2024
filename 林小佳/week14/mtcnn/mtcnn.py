import cv2
import numpy as np
from mtcnn import mtcnn

img = cv2.imread('img/test1.jpg')

model = mtcnn()  # 实例化mtcnn对象，用于后续的人脸检测
threshold = [0.5,0.6,0.7]  # 三段网络的置信度阈值不同
rectangles = model.detectFace(img, threshold)  # 使用mtcnn模型的detectFace方法检测图像中的人脸，返回检测到的人脸矩形框列表
draw = img.copy()  # 复制原始图像到变量draw中，用于后续绘制矩形框和关键点

for rectangle in rectangles:
    if rectangle is not None:  # 如果当前矩形框不为空、即检测到人脸
        W = -int(rectangle[0]) + int(rectangle[2])
        H = -int(rectangle[1]) + int(rectangle[3])
        paddingH = 0.01 * W  # 计算高度方向上的内边距，为宽度的1%
        paddingW = 0.02 * H
        crop_img = img[int(rectangle[1]+paddingH):int(rectangle[3]-paddingH), int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]  # 根据矩形框和边距裁剪图像
        if crop_img is None:  # 如果裁剪后的图像为空，则跳过当前循环迭代
            continue
        if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:  # 如果裁剪后的图像高度或宽度小于0，则跳过当前循环迭代
            continue
        cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 1)

        # 绘制检测到的人脸关键点
        for i in range(5, 15, 2):  # 遍历关键点坐标
            cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))

cv2.imwrite("img/out.jpg",draw)

cv2.imshow("test", draw)
c = cv2.waitKey(0)
