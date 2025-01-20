import torch
import torchvision.transforms as transforms 
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import numpy as np

# 加载COCO数据集上的预训练的权重
model = fasterrcnn_resnet50_fpn(pretrained = True)
# 评估模式, 即模型推理
model.eval()

device = torch.device('cpu') 
if torch.cuda.is_available():
    device = torch.device('cuda')
# 将模型移动到选定的设备（GPU 或 CPU）
model = model.to(device)

# 加载图像进行预处理, 转成tensor格式
def preprocess_image(image):
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image).unsqueeze(0) # 添加batch维度

# 进行推理
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
        if score > 0.5: # 阈值可根据需要调整
            top_left = (box[0], box[1])
            bottom_right = (box[2], box[3])
            draw.rectangle([top_left, bottom_right], outline='red', width=2)
            draw.text((box[0], box[1] - 10), str(label), fill='red')
    image.show()

# 使用示例
image_path = './street.jpg'
prediction = infer(image_path)
image = Image.open(image_path)
image = show_result(image, prediction)