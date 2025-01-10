import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 如果有GPU，加载到GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 加载预训练的Faster R-CNN模型
model = models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.to(device)
print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))

# 设置为评估模式
model.eval()

# 定义图像预处理转换
preprocess = transforms.Compose([
    transforms.ToTensor(),
])

# 加载并预处理图像
image_path = 'street.jpg'
image = Image.open(image_path).convert("RGB")
image_tensor = preprocess(image)
image_tensor = image_tensor.to(device)

# 由于模型期望输入一个列表，即使只有一个元素
images = [image_tensor]

# 模型推理
with torch.no_grad():
    predictions = model(images)
print(predictions)
# 解析预测结果
scores = predictions[0]['scores'].cpu().numpy()
boxes = predictions[0]['boxes'].cpu().numpy()
labels = predictions[0]['labels'].cpu().numpy()

# 可视化检测结果
def plot_prediction(image, boxes, labels, scores, score_threshold=0.8):
    fig, ax = plt.subplots(1)
    ax.imshow(np.array(image))
    for box, label, score in zip(boxes, labels, scores):
        if score > score_threshold:
            rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(box[0], box[1], f'{label}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))
    plt.show()

plot_prediction(image, boxes, labels, scores)
