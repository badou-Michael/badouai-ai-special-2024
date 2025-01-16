import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import numpy as np
import cv2

# 加载预训练模型
model = maskrcnn_resnet50_fpn(pretrained=True)  # 预训练
model.eval()  # 评估模式，不开启如Dropout随机丢弃神经元等操作，使用完整网络进行预测

if torch.cuda.is_available():
    device = torch.device('cuda')  # 使用NVIDIA GPU
else:
    device = torch.device('cpu')
model = model.to(device)    # 应用到模型

def preprocess_image(image):
    """
    图像预处理，image -> tensor -> 添加batch维度
    :param image: 输入图像
    :return: 优化后的tensor，[1, C, H, W]
    """
    transform = torchvision.transforms.Compose([ # torchvision.transforms.Compose：按顺序组合多个转换操作
        torchvision.transforms.ToTensor(),  # 将PIL图像转换为Tensor并归一化到[0, 1]
    ])

    return transform(image).unsqueeze(0)

def infer(path):
    """
    加载图像并进行预测
    :param path:
    :return:
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)  # 预处理
    image_tensor = image_tensor.to(device)  # 用CPU加载tensor

    with torch.no_grad():  # 关闭梯度计算，节省内存
        prediction = model(image_tensor)

    return prediction


def show_result(image, predictions):
    """
        结果可视化，绘制预测的边框和信息
        :param image:
        :param prediction:
        :return:
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color_mapping = {
        1: (255, 0, 0),  # 人用蓝色表示
        2: (0, 255, 0),  # 自行车用绿色表示
        3: (0, 0, 255)  # 汽车用红色表示
    }
    for pred in predictions:
        masks = pred['masks'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        for mask, label, score in zip(masks, labels, scores):
            if score > 0.5:
                mask = mask[0]
                mask = (mask > 0.5).astype(np.uint8)
                color = color_mapping.get(label.item(), (255, 255, 255))
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, color, 2)
    image = cv2.resize(image, (700, 700))
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = 'street.jpg'

# 推理
prediction = infer(image_path)

# 读取原始图像
image = Image.open(image_path)

# 显示结果
image = show_result(image, prediction)

