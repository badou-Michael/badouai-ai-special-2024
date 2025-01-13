import torch
import cv2
import numpy as np
from nets.mrcnn import MaskRCNN
from utils.config import Config
from utils.visualize import display_instances
import torchvision
import torchvision.transforms as T
import os
import urllib.request

# COCO类别名称
COCO_CLASSES = [
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
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def predict(weights_path, config=None):
    if config is None:
        config = Config()
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到权重文件: {weights_path}")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 使用torchvision预训练模型
    print("加载预训练模型...")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = model.to(device)
    model.eval()
    
    # 读取图片
    print(f"读取图片: {weights_path}")
    image = cv2.imread(weights_path)
    if image is None:
        raise ValueError(f"无法读取图片: {weights_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 预处理
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).to(device)
    
    # 推理
    print("执行推理...")
    with torch.no_grad():
        prediction = model([image_tensor])[0]
    
    # 后处理
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    masks = prediction['masks'].squeeze(1).cpu().numpy()
    labels = [COCO_CLASSES[i] if i < len(COCO_CLASSES) else 'unknown' 
             for i in prediction['labels'].cpu().numpy()]
    
    print(f"检测到 {len(boxes)} 个目标")
    
    # 可视化
    print("显示结果...")
    display_instances(image, boxes, masks, scores, COCO_CLASSES)

if __name__ == '__main__':
    try:
        weights_path = "data/street.jpg"
        predict(weights_path)
    except Exception as e:
        print(f"错误: {str(e)}") 