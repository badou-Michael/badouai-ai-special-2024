#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：maskrcnn.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2025/1/8 15:55 
'''

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

"""
Mask R-CNN实例分割实现
该程序使用预训练的Mask R-CNN模型进行图像实例分割
可以检测和分割图像中的多个对象，并为每个对象生成像素级掩码
"""

# 初始化Mask R-CNN模型
"""
Mask R-CNN模型架构说明：
1. 基础网络：使用ResNet50作为特征提取器
2. FPN（特征金字塔网络）：用于处理多尺度目标检测
3. COCO预训练权重：模型在COCO数据集上预训练，可以识别80种常见物体
"""
model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)

# 将模型设置为评估模式，这会禁用如dropout等训练时使用的特性
model.eval()

# 设置计算设备：优先使用GPU，如果没有GPU则使用CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)


def preprocess_image(image):
    """
    图像预处理的关键步骤：
    1. ToTensor():
       - 将PIL图像或numpy数组转换为torch张量
       - 将像素值范围从[0,255]缩放到[0,1]
       - 调整维度顺序从(H,W,C)到(C,H,W)
    2. unsqueeze(0): 添加batch维度，最终维度为(1,C,H,W)
    这些步骤是深度学习模型处理图像的标准要求

    Args:
        image: PIL格式的输入图像
    Returns:
        tensor: 预处理后的图像张量
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


def infer(image_path):
    """
    推理过程说明：
    1. 图像加载和预处理
    2. 使用torch.no_grad()进行推理
       - 禁用梯度计算，减少内存使用
       - 加快推理速度
    3. 模型输出包含多个字典，每个字典包含：
       - boxes: 边界框坐标
       - labels: 类别标签
       - scores: 置信度分数
       - masks: 实例分割掩码

    Args:
        image_path: 输入图像的路径
    Returns:
        prediction: 模型的预测结果
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    return prediction


def show_result(image_path, predictions):
    """
    可视化实例分割结果

    Args:
        image_path: 原始图像路径
        predictions: 模型预测结果，包含masks、labels和scores
    """
    # 读取原始图像并转换为RGB格式
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 用于存储每个实例的随机颜色
    instance_colors = {}

    for pred in predictions:
        # 获取模型预测的三个关键组件
        masks = pred['masks'].cpu().numpy()  # 实例分割掩码
        labels = pred['labels'].cpu().numpy()  # 类别标签
        scores = pred['scores'].cpu().numpy()  # 置信度分数

        """
        预测结果解析：
        - masks: 形状为[N,1,H,W]的数组，N是检测到的实例数量
        - labels: 长度为N的数组，包含每个实例的类别ID
        - scores: 长度为N的数组，包含每个预测的置信度
        """

        for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
            if score > 0.5:  # 置信度阈值过滤
                # 掩码处理
                mask = mask[0]  # 获取单个实例的掩码
                """
                掩码处理说明：
                1. mask[0]: 取出第一个通道（因为掩码是单通道的）
                2. 二值化: 将概率值转换为二进制掩码
                   - >0.5的像素认为属于该实例
                   - ≤0.5的像素认为是背景
                """
                mask = (mask > 0.5).astype(np.uint8)  # 二值化处理

                # 为每个实例生成唯一的随机颜色
                if i not in instance_colors:
                    instance_colors[i] = (
                        np.random.randint(0, 256),
                        np.random.randint(0, 256),
                        np.random.randint(0, 256)
                    )
                color = instance_colors[i]

                # 轮廓检测和绘制
                """
                轮廓检测原理：
                1. cv2.RETR_TREE: 检测所有轮廓并重建层次结构
                2. cv2.CHAIN_APPROX_SIMPLE: 压缩水平、垂直和对角线段，只保留端点
                """
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # 在原图上绘制轮廓
                """
                绘制参数说明：
                - image: 目标图像
                - contours: 轮廓点集
                - -1: 绘制所有轮廓
                - color: 轮廓颜色
                - 2: 轮廓线条粗细
                """
                cv2.drawContours(image, contours, -1, color, 2)

    # 调整图像大小并显示
    image = cv2.resize(image, (700, 700))
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    image_path = 'street.jpg'  # 设置输入图像路径
    prediction = infer(image_path)  # 进行实例分割
    image = Image.open(image_path)  # 打开原始图像
    show_result(image_path, prediction)  # 显示分割结果