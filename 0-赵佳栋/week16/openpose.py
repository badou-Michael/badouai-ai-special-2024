#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：openpose.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2025/1/15 13:07
'''
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np


def load_model():
    """
    加载预训练的OpenPose模型
    返回:
        model: 预训练好的OpenPose模型实例
    """
    model = torch.hub.load('CMU-Visual-Computing-Lab/openpose',
                           'pose_resnet50',
                           pretrained=True)
    model.eval()  # 设置为评估模式
    return model


def preprocess_image(image):
    """
    图像预处理函数
    参数:
        image: numpy.ndarray, 输入图像
    返回:
        tensor: torch.Tensor, 预处理后的图像张量
    """
    # 定义图像转换流程
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        # 标准化处理，使用ImageNet的均值和标准差
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])
    # 转换图像并添加batch维度
    return transform(image).unsqueeze(0)


def detect_keypoints(image, model):
    """
    检测图像中的关键点
    参数:
        image: numpy.ndarray, 输入图像
        model: torch.nn.Module, OpenPose模型
    返回:
        image: numpy.ndarray, 标注了关键点的图像
        keypoints: numpy.ndarray, 检测到的关键点坐标
    """
    # 图像预处理
    image_tensor = preprocess_image(image)

    # 模型推断
    with torch.no_grad():
        output = model(image_tensor)

    # 获取热图
    heatmaps = output[0].cpu().numpy()

    # 从热图中提取关键点
    keypoints = np.zeros((heatmaps.shape[0], 2))
    for i in range(heatmaps.shape[0]):
        # 获取每个关键点的最大响应位置
        y, x = np.unravel_index(np.argmax(heatmaps[i]),
                                heatmaps[i].shape)
        keypoints[i] = [x, y]
        # 在图像上绘制关键点
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

    return image, keypoints


def main():
    """
    实现人体姿态估计的完整流程
    """
    # 加载模型
    model = load_model()

    # 读取图像
    image_path = "demo.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
        return

    # 检测关键点
    result_image, keypoints = detect_keypoints(image, model)

    # 显示结果
    cv2.imshow('Human Pose Estimation', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()