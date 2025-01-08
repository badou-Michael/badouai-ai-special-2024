import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 加载预训练模型
# model = maskrcnn_resnet50_fpn(pretrained=True)
model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)  # 使用在COCO数据集上预训练的ResNet-50 FPN模型的权重

model.eval()

# 如果你的模型是在GPU上训练的，确保模型也在GPU上进行推理
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)


# 加载图像并进行预处理
def preprocess_image(image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # 添加batch维度


# 进行推理
def infer(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    return prediction


# 显示结果
def show_result(image_path, predictions):
    image = cv2.imread(image_path)
    # 将图像从BGR格式转换为RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    instance_colors = {}
    for pred in predictions:
        # 获取预测结果中的掩码（masks）
        masks = pred['masks'].cpu().numpy()
        # 获取预测结果中的标签（labels）
        labels = pred['labels'].cpu().numpy()
        # 获取预测结果中的置信度分数（scores）
        scores = pred['scores'].cpu().numpy()
        for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
            if score > 0.5:
                # 取出掩码数组的第一个元素
                mask = mask[0]
                # 将掩码转换为二值图像，其中值大于0.5的像素设置为1，其余设置为0
                mask = (mask > 0.5).astype(np.uint8)
                if i not in instance_colors:
                    instance_colors[i] = (
                        # 如果当前实例索引i不在instance_colors字典中，则为其生成一个随机颜色
                        np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                # 获取当前实例的颜色
                color = instance_colors[i]
                # 使用OpenCV的findContours函数检测掩码中的轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # 在图像上绘制轮廓
                cv2.drawContours(image, contours, -1, color, 2)

    image = cv2.resize(image, (700, 700))
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 使用示例
image_path = 'street.jpg'  # 替换为你的图像路径
prediction = infer(image_path)
image = Image.open(image_path)
show_result(image_path, prediction)
