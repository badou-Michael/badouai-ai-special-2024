# -*- coding: utf-8 -*-
# time: 2024/11/26 23:53
# file: detect_Face.py
# author: flame
import cv2
from mtcnn import mtcnn

''' 该脚本使用 MTCNN 模型检测图像中的人脸，并在检测到的人脸周围绘制矩形框和关键点。 '''

''' 读取图像文件 "img/timg.jpg" 并将其存储在变量 img 中。 '''
img = cv2.imread("img/timg.jpg")

''' 初始化 MTCNN 模型实例。 '''
model = mtcnn()

''' 设置人脸检测的阈值，分别为三个阶段的阈值。 '''
threshold = [0.5, 0.6, 0.7]

''' 使用 MTCNN 模型检测图像中的人脸，返回检测到的人脸矩形框列表。 '''
rectangles = model.detectFace(img, threshold)

''' 创建图像的副本，以便在副本上绘制检测结果，避免修改原始图像。 '''
img_copy = img.copy()

''' 遍历检测到的每个人脸矩形框。 '''
for rectangle in rectangles:
    ''' 检查当前矩形框是否为空。 '''
    if rectangle is not None:
        ''' 计算矩形框的宽度 W。 '''
        W = -int(rectangle[0]) + int(rectangle[2])
        ''' 计算矩形框的高度 H。 '''
        H = -int(rectangle[1]) + int(rectangle[3])
        ''' 计算矩形框的垂直填充量，占宽度的 1%。 '''
        paddingH = 0.01 * W
        ''' 计算矩形框的水平填充量，占高度的 1%。 '''
        paddingW = 0.01 * H
        ''' 根据矩形框和填充量裁剪图像。 '''
        crop_img = img[int(rectangle[1] - paddingH):int(rectangle[3] + paddingH), int(rectangle[0] - paddingW):int(rectangle[2] + paddingW)]
        ''' 检查裁剪后的图像是否为空。 '''
        if crop_img is None:
            ''' 如果裁剪后的图像为空，打印提示信息并跳过当前循环。 '''
            print("crop_img is None")
            continue
        ''' 检查裁剪后的图像尺寸是否有效。 '''
        if crop_img.shape[0] < 0 and crop_img.shape[1] < 0:
            ''' 如果裁剪后的图像尺寸无效，跳过当前循环。 '''
            continue
        ''' 在图像副本上绘制矩形框，表示检测到的人脸位置。 '''
        cv2.rectangle(img_copy, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (0, 255, 0), 1)

        ''' 遍历人脸的关键点，并在图像副本上绘制这些关键点。 '''
        for i in range(5, 15, 2):
            ''' 在图像副本上绘制关键点，颜色为蓝色。 '''
            cv2.circle(img_copy, (int(rectangle[i] + 0), int(rectangle[i + 1])), 2, (255, 0, 0))

''' 将处理后的图像保存为 "img/timg_result.jpg"。 '''
cv2.imwrite("img/timg_result.jpg", img_copy)

''' 显示处理后的图像。 '''
cv2.imshow("img", img_copy)

''' 等待用户按键，按任意键关闭图像窗口。 '''
c = cv2.waitKey(0)
