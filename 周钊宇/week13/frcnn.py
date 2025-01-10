import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F 
import numpy as np 
import cv2
from PIL import ImageDraw,Image


#加载模型
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()   #模型推理


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)


#加载图像预处理

def preprocess(image):
    tranfroms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    return tranfroms(image).unsqueeze(0) #在第0维度增加一个维度

def infer(img_path):
    image = Image.open(img_path).convert('RGB')
 
    image_tensor = preprocess(image)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        predict = model(image_tensor)

    return predict

def show_res(image, prediction):
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    draw = ImageDraw.Draw(image)

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            top_left = (box[0], box[1])
            bottom_right = (box[2], box[3])
            draw.rectangle([top_left, bottom_right], outline='red', width=2)
            draw.text((box[0], box[1] - 10), str(label), fill='red')
    image.show()

image_path = 'test.jpeg'
predict = infer(image_path)
image = Image.open(image_path)
image = show_res(image, predict)