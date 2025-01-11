#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/12/28 11:38
# @Author: Gift
# @File  : torch-fasterRcnn.py
# @IDE   : PyCharm
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image,ImageDraw
#加载预训练的模型
model = fasterrcnn_resnet50_fpn(pretrained=True)
#将模型设置为评估模式
model.eval()
#确定设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#将模型加载到指定设备
model = model.to(device)
# 加载图像并进行预处理
def preprocess_image(image):
    transform = torchvision.transforms.Compose([    #组合一系列的图像变换操作
        torchvision.transforms.ToTensor(), # 将图像转换为PyTorch张量，并归一化到[0, 1]范围
    ])
    return transform(image).unsqueeze(0)  # 将单张图像的张量转换为一个大小为 (1, C, H, W) 的张量
#进行推理
def infer(image_path):
    image = Image.open(image_path).convert("RGB")  #方法将图像转换为 RGB 格式，确保图像具有三个颜色通道，这是很多图像模型所期望的输入格式。
    image_tensor = preprocess_image(image) #调用preprocess_image函数将图像转换为模型所需的张量格式
    image_tensor = image_tensor.to(device) #将图像张量移动到指定的设备（GPU或CPU）上

    with torch.no_grad(): #推理不需要 梯度
        prediction = model(image_tensor)

    return prediction
#显示结果
def show_result(image, prediction):
    #从预测中提取边界框，标签，和得分
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    draw = ImageDraw.Draw(image)

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # 阈值可根据需要调整
            top_left = (box[0], box[1])
            bottom_right = (box[2],box[3])
            draw.rectangle([top_left, bottom_right], outline='red', width=2) #制矩形边界框
            print(str(label))
            draw.text((box[0], box[1] - 10), str(label), fill='red')
    image.show()
if __name__ == '__main__':
    image_path = 'street.jpg'  # 替换为你的图像路径
    prediction = infer(image_path)
    print(prediction)
    image = Image.open(image_path)
    image = show_result(image, prediction)
