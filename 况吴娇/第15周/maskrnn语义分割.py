import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn ##使用maskrcnn_resnet50_fpn函数加载一个预训练的Mask R-CNN模型，该模型使用ResNet-50作为骨干网络，并采用特征金字塔网络（FPN）进行特征提取。
from torchvision.transforms import functional as F
from PIL import Image,ImageDraw #从PIL库中导入Image和ImageDraw模块，用于图像的加载和绘制.
import numpy as np
import cv2


# 加载预训练模型
model = maskrcnn_resnet50_fpn(pretrained=True)
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
        prediction = model(image_tensor) #with torch.no_grad(): prediction = model(image_tensor) 在不计算梯度的情况下进行推理，以提高效率.

    return prediction

# 显示结果
def show_result(image, predictions):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color_mapping = {
        1: (255, 0, 0),  # 人用蓝色表示
        2: (0, 255, 0),  # 自行车用绿色表示
        3: (0, 0, 255)   # 汽车用红色表示
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
    image=cv2.resize(image,(700,700))
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用示例
image_path = 'street.jpg'  # 替换为你的图像路径
prediction = infer(image_path)
image = Image.open(image_path)
image = show_result(image, prediction)

'''
label.item() 将标签从一个 PyTorch 张量转换为一个 Python 原生整数。label 是一个包含标签的张量，item() 方法将其转换为一个普通的整数，以便可以在字典中查找。
color_mapping.get(label.item(), (255, 255, 255)) 使用字典的 get 方法获取标签对应的颜色。get 方法的参数如下：
label.item()：键，即标签的整数值。
(255, 255, 255)：默认值，如果字典中没有找到对应的键，则返回这个默认值。这里默认值是白色。



contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
这行代码的目的是找到掩码中的轮廓。具体解释如下：
mask 是一个二值图像，其中每个像素值要么是 0（背景），要么是 1（前景）。这个掩码表示了目标在图像中的位置。
cv2.findContours 是 OpenCV 中的一个函数，用于找到二值图像中的轮廓。函数的参数如下：
mask：输入的二值图像。
cv2.RETR_TREE：轮廓检索模式，表示构建一个轮廓的层级树。这意味着可以找到轮廓的父子关系。
cv2.CHAIN_APPROX_SIMPLE：轮廓近似方法，表示使用简单的链式近似算法来压缩轮廓，减少轮廓的点数，从而提高效率。
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 的返回值是一个元组，包含两个元素：
contours：一个列表，包含找到的轮廓。每个轮廓是一个 NumPy 数组，表示轮廓的点集。
_：轮廓的层级信息，通常在处理复杂的轮廓结构时使用。这里我们不需要这个信息，所以用 _ 忽略它。
'''
