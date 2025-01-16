#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：predict.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2025/1/16 16:16 
'''

import glob
import numpy as np
import torch
import os
import cv2
from unet_model import UNet


def load_model(model_path, device):
    """
    加载预训练的UNet模型
    参数:
        model_path (str): 模型权重文件路径
        device (torch.device): 运行设备
    返回:
        model: 加载好权重的模型
    """
    # 初始化UNet模型（单通道输入，单类别输出）
    model = UNet(n_channels=1, n_classes=1)
    # 将模型移至指定设备
    model.to(device=device)
    # 加载预训练权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    # 设置为评估模式
    model.eval()
    return model


def preprocess_image(image_path):
    """
    图像预处理
    参数:
        image_path (str): 图像路径
    返回:
        tensor: 预处理后的图像张量
    """
    # 读取图像
    img = cv2.imread(image_path)
    # 转换为灰度图
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 重塑维度为 [1, 1, height, width]
    img = img.reshape(1, 1, img.shape[0], img.shape[1])
    return img


def process_prediction(pred):
    """
    处理模型预测结果
    参数:
        pred (torch.Tensor): 模型预测输出
    返回:
        numpy.ndarray: 处理后的二值图像
    """
    # 转换为numpy数组
    pred = np.array(pred.data.cpu()[0])[0]
    # 二值化处理
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0
    return pred


def main():
    """
    实现图像分割的推理过程
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model_path = 'best_model.pth'
    net = load_model(model_path, device)

    # 获取测试图像路径
    tests_path = glob.glob('data/test/*.png')
    print(f"Found {len(tests_path)} test images")

    # 处理每张测试图像
    for test_path in tests_path:
        try:
            # 生成结果保存路径
            save_res_path = test_path.split('.')[0] + '_res.png'

            # 图像预处理
            img = preprocess_image(test_path)

            # 转换为tensor并移至指定设备
            img_tensor = torch.from_numpy(img)
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)

            # 模型推理
            with torch.no_grad():
                pred = net(img_tensor)

            # 处理预测结果
            result = process_prediction(pred)

            # 保存结果
            cv2.imwrite(save_res_path, result)
            print(f"Processed: {test_path}")

        except Exception as e:
            print(f"Error processing {test_path}: {str(e)}")


if __name__ == "__main__":
    main()