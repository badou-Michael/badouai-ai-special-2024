import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import numpy as np

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)


def detect_objects(image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)


def infer(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = detect_objects(image)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        detections = model(image_tensor)
    return detections


def image_result_show(image, detections):
    boxes = detections[0]['boxes'].cpu().numpy()
    labels = detections[0]['labels'].cpu().numpy()
    scores = detections[0]['scores'].cpu().numpy()
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            top_left = (box[0], box[1])
            bottom_right = (box[2], box[3])
            draw.rectangle([top_left, bottom_right], outline='red', width=2)
            print(str(label))
            draw.text((box[0], box[1] - 10), str(label), fill='red')
        image.show()


image_path = 'strawberry.jpg'
detections = infer(image_path)
image = Image.open(image_path)
image_result_show(image, detections)
print(image_result_show)
