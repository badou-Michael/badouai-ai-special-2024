#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@FileName    :marskrccn.py
@Time    :2025/01/09 09:39:21
@Author    :chungrae
@Description: marskrccn 实现

'''

from pathlib import Path

import numpy as np

import cv2

import torch
from torch import Tensor
import torch.cuda
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

from PIL import Image


class MarskRcnn:
    
    def __init__(self, image_fp: Path):
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch("cpu")
        self.model = self.model.to(self.device)
        
        self.image = Image.open(image_fp).convert("RGB")
        
    def processing(self) -> Tensor:
        """图片预处理
        :return: 图片张量
        """
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        return transform(self.image).unsqueeze(0)  # 添加batch维度
    
    
    def infer(self) -> Tensor:
        """推理"""
       
        img_tensor = self.processing()
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            prediction = self.model(img_tensor)
            
        return prediction

    def show(self):
        """展示推理结果

        :param img: 图片信息
        :param predictions: 推理结果
        """
        predictions = self.infer()
        colors = {}
        for predction in predictions:
            masks = predction["masks"].cpu().numpy()
            labels = predction["labels"].cpu().numpy()
            scores = predction["scores"].cpu().numpy()

            for i, mask, _, score in enumerate(zip(masks, labels, scores)):
                if score > 0.5:
                    mask = mask[0]
                    mask = (mask > 0.5).astype(np.int8)
                    if i not in colors:
                        colors[i] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                    color = colors[i]
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(self.image,contours, -1, color, 2)
                    
            cv2.imgShow("result", self.image)
            cv2.waitKey(0)
            cv2.destoryAllWindows()
            
            
if __name__ == "__main__":
    image_filepath = Path("./street.jpg")
    
    model = MarskRcnn(Path("./street.jpg"))
    
    model.show()
    