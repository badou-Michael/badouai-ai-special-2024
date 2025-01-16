'''
Mask R-CNN 是一个两阶段的框架，第一个阶段扫描图像并生成提议（proposals，即有可能包含一个目标的区域），第二阶段分类提议并生成边界框和掩码。
算法步骤:
首先，输入一幅你想处理的图片，然后进行对应的预处理操作，或者预处理后的图片；
然后，将其输入到一个预训练好的神经网络中（ResNeXt等）获得对应的feature map；
接着，对这个feature map中的每一点设定预定个的ROI，从而获得多个候选ROI；
接着，将这些候选的ROI送入RPN网络进行二值分类（前景或背景）和BB回归，过滤掉一部分候选的ROI；
接着，对这些剩下的ROI进行ROIAlign操作（即先将原图和feature map的pixel对应起来，然后将feature map和固定的feature对应起来）；
最后，对这些ROI进行分类（N类别分类）、BB回归和MASK生成（在每一个ROI里面进行FCN操作）。

'''

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn  #调用
from torchvision.transforms import functional as F
from PIL import Image,ImageDraw
import numpy as np
import cv2

model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

def preprocess_image(image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0) #增加batch维度

def infer(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction

def show(image, predictions):
    image = cv2.imread(image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    color_mapping = {
        1: (255, 0, 0),  # 人-蓝色
        2: (0, 255, 0),  # 自行车-绿色
        3: (0, 0, 255)  # 汽车-红色
    }
    for pred in predictions:
        masks = pred['masks'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        for mask, label, score in zip(masks, labels, scores):
            if score > 0.5:
                mask = mask[0]
                mask = (mask > 0.5).astype(np.uint8)
                color = color_mapping.get(label.item(), (255, 255, 255))
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, color, 2)

    image=cv2.resize(image,(700,700))
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = '图像分割/street.jpg'  # 图像路径
prediction = infer(image_path)
image = Image.open(image_path)
image = show(image, prediction)
