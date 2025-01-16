#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/1/8 21:48
@Author  : Mr.Long
@Content : 图像分割：简单版
"""
import os.path

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import numpy as np
import cv2
from common.path import train_test
import os


class HomeworkW15MaskRCNN:
    def __init__(self):
        # 加载预训练模型
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        # 如果你的模型是在GPU上训练的，确保模型也在GPU上进行推理
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.model.to(self.device)

    # 加载图像并进行预处理
    @staticmethod
    def preprocess_image(img):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        return transform(img).unsqueeze(0)  # 添加batch维度

    # 进行推理
    def infer(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.preprocess_image(img)
        image_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            predict = self.model(image_tensor)
        return predict

    # 显示结果
    @staticmethod
    def show_result(img_path, predictions):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        instance_colors = {}
        for pred in predictions:
            masks = pred['masks'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
                if score > 0.5:
                    mask = mask[0]
                    mask = (mask > 0.5).astype(np.uint8)
                    if i not in instance_colors:
                        instance_colors[i] = (np.random.randint(0, 256), np.random.randint(0, 256),
                                              np.random.randint(0, 256))
                    color = instance_colors[i]
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, contours, -1, color, 2)
        cv2.imshow('Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':

    image_path = os.path.join(train_test, 'mask_rcnn\\street.jpg')
    w15 = HomeworkW15MaskRCNN()
    # 使用示例
    prediction = w15.infer(image_path)
    image = Image.open(image_path)
    w15.show_result(image, prediction)

