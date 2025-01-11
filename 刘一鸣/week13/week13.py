'''
week13作业：
实现fasterrcnn（调接口or手写）
'''

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((400, 400)),  # VOC 图像尺寸
])

# 加载 PASCAL VOC 数据集
train_dataset = datasets.VOCDetection(
    root='./data',
    year='2007',
    image_set='train',
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# 加载 Faster R-CNN 预训练模型
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

# 替换分类头
num_classes = 21  # VOC 有 20 类，加背景类共 21 类
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor =  FastRCNNPredictor(in_features, num_classes)

# 设置设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 优化器和学习率调度器
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# 数据集的类名称
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    'tvmonitor'
]

# 训练函数
def train(model, train_loader, optimizer, lr_scheduler, device):
    model.train()
    for images, targets in train_loader:
        images = [image.to(device) for image in images]
        processed_targets = []
        for target in targets:
            boxes = []
            labels = []
            for obj in target['annotation']['object']:
                class_name = obj['name']
                if class_name not in VOC_CLASSES:
                    continue
                bbox = obj['bndbox']
                xmin, ymin, xmax, ymax = float(bbox['xmin']), float(bbox['ymin']), float(bbox['xmax']), float(bbox['ymax'])
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(VOC_CLASSES.index(class_name))
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([0]),
                'area': torch.tensor([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]),
                'iscrowd': torch.tensor([0] * len(labels))
            }
            processed_targets.append(target)
        loss_dict = model(images, processed_targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    lr_scheduler.step()

# 训练模型
for epoch in range(10):
    print(f"Epoch {epoch + 1}")
    train(model, train_loader, optimizer, lr_scheduler, device)

# 保存模型
torch.save(model.state_dict(), 'fasterrcnn_voc.pth')

# 预测和推理部分
def inference(model, image_path):
    model.eval()  # 设置为评估模式
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # 转换为批次格式

    with torch.no_grad():
        prediction = model(input_tensor.to(device))[0]  # 获取第一个预测结果

    draw = ImageDraw.Draw(image)
    for i, score in enumerate(prediction['scores']):
        if score > 0.5:  # 阈值设定，置信度大于 0.5 进行绘制
            box = prediction['boxes'][i].cpu().numpy()
            label = VOC_CLASSES[prediction['labels'][i].item()]
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), f"{label}: {score:.2f}", fill="red")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# 进行推理
inference(model, './street.jpg')  # 替换为测试图片路径

