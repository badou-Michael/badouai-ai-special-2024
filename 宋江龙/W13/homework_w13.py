#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/26 22:30
@Author  : Mr.Long
@Content : 第13周作业
"""
import os.path

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
from common.path import fasterrcnn

class HomeworkW13FasterRCNN:

    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn()
        self.model.eval()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)

    # 加载图像并进行预处理
    @staticmethod
    def preprocess_image_w13(image):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
        return transform(image).unsqueeze(0)

    def infer_w13(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess_image_w13(image)
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            prediction = self.model(image_tensor)
        return prediction

    @staticmethod
    def show_result_w13(image, prediction):
        boxes = prediction[0]['boxes'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        draw = ImageDraw.Draw(image)
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5:
                top_left = (box[0], box[1])
                bottom_right = (box[2], box[3])
                draw.rectangle([top_left, bottom_right], outline='red', width=2)
                print(str(label))
                draw.text((box[0], box[1] - 10), str(label), fill='red')
        image.show()



if __name__ == '__main__':
    homework = HomeworkW13FasterRCNN()
    image_path = os.path.join(fasterrcnn, 'street.jpg')  # 替换为你的图像路径
    prediction = homework.infer_w13(image_path)
    image = Image.open(image_path)
    image = homework.show_result_w13(image, prediction)


