'''
Faster RCNN模型可以分为四个模块
1.Conv layers，特征提取网络
输入为一张图片，输出为一张图片的特征，即feature map。通过一组conv+relu+pooling层提取图像的feature map，用于后续的RPN网络和全连接层。
2.Region proposal Network，区域候选网络
输入为第一步中的feature map，输出为多个兴趣区域（ROI）。输出的每个兴趣区域具体表示为一个概率值（用于判断anchor是前景还是背景）和四个坐标
值，概率值表示该兴趣区域有物体的概率，这个概率是通过softmax对每个区域进行二分类得到的；坐标值是预测的物体的位置，在进行训练时会用这个坐标与
真实的坐标进行回归使在测试时预测的物体位置更加准确。
3.ROI pooling，兴趣域池化
这一层以RPN网络输出的兴趣区域和Conv layers输出的feature map为输入，将两者进行综合后得到固定大小的区域特征图（proposal feature map）
并输出到后面的全连接网络中进行分类。
4.Classification and Regression，分类和回归
输入为上一层得到proposal feature map，输出为兴趣区域中物体所属的类别以及物体在图像中精确的位置。这一层通过softmax对图像进行分类，并通
过边框回归修正物体的精确位置。
'''
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image,ImageDraw
import numpy as np

# 加载预训练模型
model = fasterrcnn_resnet50_fpn(pretrained = True)
model.eval()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# jiazaitupian
def preprocess_image(image):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    return transform(image).unsqueeze(0)

# 推理
def infer(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction

# 显示结果
def show_result(image, prediction):
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    draw = ImageDraw.Draw(image)

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.4:  # 阈值,可调整
            top_left = (box[0], box[1])
            bottom_right = (box[2],box[3])
            draw.rectangle([top_left, bottom_right], outline='red', width=2)
            print(str(label))
            draw.text((box[0], box[1] - 10), str(label), fill='red')
    image.show()

# 使用示例
image_path = 'D:\算法学习\CV课程1\street.jpg'  # 图像路径
prediction = infer(image_path)
image = Image.open(image_path)
image = show_result(image, prediction)
