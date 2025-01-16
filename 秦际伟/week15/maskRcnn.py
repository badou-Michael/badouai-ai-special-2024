import torch
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 检查是否有可用的 GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 加载预训练的 Mask R-CNN 模型并将其移动到设备上
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
model.to(device)
model.eval()

# 获取 COCO 类别标签
coco_class_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# 加载测试图像
image_path = "street.jpg"
image = Image.open(image_path).convert("RGB")
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# 将图像转换为 tensor 并移动到设备上
transform = T.Compose([T.ToTensor()])
image_tensor = transform(image).to(device)

# 进行预测
with torch.no_grad():
    prediction = model([image_tensor])

# 解析预测结果
boxes = prediction[0]['boxes'].cpu().numpy()
labels = prediction[0]['labels'].cpu().numpy()
scores = prediction[0]['scores'].cpu().numpy()
masks = prediction[0]['masks'].cpu().numpy()

# 设置阈值过滤低置信度的预测结果
threshold = 0.5
filtered_boxes = boxes[scores > threshold]
filtered_labels = labels[scores > threshold]
filtered_masks = masks[scores > threshold].squeeze(1) > 0.5

# 定义一个函数来生成随机颜色
def generate_random_color():
    return tuple(np.random.randint(0, 256, size=3).tolist())

# 绘制结果
for box, label, mask in zip(filtered_boxes, filtered_labels, filtered_masks):
    # 获取类别名称
    class_name = coco_class_names[label]

    # 为每个实例生成一个唯一的随机颜色
    color = generate_random_color()

    # 绘制边界框
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)

    # 绘制标签和置信度
    label_text = f'{class_name} {scores[np.where(labels == label)][0]:.2f}'
    cv2.putText(original_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 绘制掩码
    colored_mask = np.zeros_like(original_image)
    colored_mask[mask] = color
    original_image = cv2.addWeighted(original_image, 1, colored_mask, 0.5, 0)

# 显示结果
plt.figure(figsize=(8, 6))
plt.imshow(original_image)
plt.axis('off')
plt.show()



