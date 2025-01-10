import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import numpy as np
import cv2

# 加载预训练模型
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 如果你的模型是在GPU上训练的，确保模型也在GPU上进行推理
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)


# 加载图像并进行预处理
def preprocess_image(p_image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return transform(p_image).unsqueeze(0)  # 添加batch维度


# 进行推理
def infer(infer_image_path):
    image = Image.open(infer_image_path).convert("RGB")       # 转换
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    return prediction


# 显示结果
def show_result(predictions):
    s_image = cv2.imread(image_path)
    s_image = cv2.cvtColor(s_image, cv2.COLOR_BGR2RGB)
    instance_colors = {
        1: (255, 0, 0),  # 人用蓝色表示
        2: (0, 255, 0),  # 自行车用绿色表示
        3: (0, 0, 255)  # 汽车用红色表示
    }
    for pred in predictions:
        masks = pred['masks'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
            if score > 0.5:
                mask = mask[0]
                mask = (mask > 0.5).astype(np.uint8)
                if i not in instance_colors:
                    instance_colors[i] = (
                    np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                color = instance_colors[i]
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(s_image, contours, -1, color, 2)

    cv2.imshow('Result', s_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 使用示例
image_path = 'street.jpg'  # 读图
prediction = infer(image_path)
image = Image.open(image_path)
show_result(prediction)
