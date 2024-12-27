import torchvision
from PIL import Image
import torch
from PIL import ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms

## 加载模型并调整为评估模式
model = fasterrcnn_resnet50_fpn(pretrained=True);
# model.train()：设置模型为训练模式（启用 Dropout 和 BatchNorm 的批次统计）。
# model.eval()：设置模型为评估模式（禁用 Dropout，BatchNorm 使用全局统计量）。
model.eval();

## 检查是否有GPU，如果没有就放到CPU运行
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


## 定义图片处理串并加上batch size
def preprocess_img(img):
    predefine = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0, ], [1, ])
        ]
    )
    return predefine(img).unsqueeze(0)


# 推理图片
def inference(img):
    img = Image.open(img).convert("RGB")
    image_tensor = preprocess_img(img)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        result = model(image_tensor)

    return result


# 展示推理结果
def show_result(img, result):
    # 获取目标检测框位置
    boxes = result[0]['boxes'].cpu().numpy()
    # 获取标签
    labels = result[0]['labels'].cpu().nunpy()
    # 获取置信度
    scores = result[0]['scores'].cpu().numpy()
    draw = ImageDraw.Draw(img)

    for box, label, scores in zip(boxes, labels, scores):
        if scores > 0.5:
            top_left = (box[0], box[1])
            bottom_right = (box[2], box[3])
            draw.rectangle([top_left, bottom_right], outline='green', width=2)
            draw.text((box[0], box[1]), str(label), fill='green')
    img.show()

image_path = 'E:/practice/八斗/课程/八斗AI2024精品班/【13】目标检测/代码/fasterrcnn简单版/street.jpg'
prediction = inference(image_path)
img = Image.open(image_path)
img = show_result(img, prediction)
