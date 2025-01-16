import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 1. 加载预训练的 Faster R-CNN 模型（包含预训练权重）
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # 设置为评估模式


# 2. 图像预处理函数
def preprocess_image(image_path):
    # 打开图像
    image = Image.open(image_path).convert("RGB")

    # 定义预处理流程，包括调整大小、转换为Tensor和标准化
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
    ])

    image_tensor = transform(image)

    # 添加批次维度
    image_tensor = image_tensor.unsqueeze(0)  # shape: (1, C, H, W)

    return image_tensor


# 3. 运行推理
def predict(image_path):
    # 预处理图像
    image_tensor = preprocess_image(image_path)

    # 判断是否有可用的GPU，如果有则使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    image_tensor = image_tensor.to(device)

    # 进行推理
    with torch.no_grad():
        predictions = model(image_tensor)

    return predictions


# 4. 处理推理结果并绘制
def draw_results(image_path, predictions):
    # 打开图像
    image = cv2.imread(image_path)

    # 获取预测框、类别和置信度
    boxes = predictions[0]['boxes'].cpu().numpy()  # shape: (num_boxes, 4)
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    # 可视化
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # 只绘制置信度大于0.5的框
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(image, f'{label}: {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 0, 0), 2)

    # 显示结果图像
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# 5. 测试推理
image_path = 'street.jpg'  # 替换为图片路径
predictions = predict(image_path)
draw_results(image_path, predictions)
