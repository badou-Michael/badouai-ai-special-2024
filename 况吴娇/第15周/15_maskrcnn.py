import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn ##使用maskrcnn_resnet50_fpn函数加载一个预训练的Mask R-CNN模型，该模型使用ResNet-50作为骨干网络，并采用特征金字塔网络（FPN）进行特征提取。
from torchvision.transforms import functional as F
from PIL import Image,ImageDraw #从PIL库中导入Image和ImageDraw模块，用于图像的加载和绘制.
import numpy as np
import cv2

# 加载预训练模型
model = maskrcnn_resnet50_fpn(pretrained=True) #创建一个Mask R-CNN模型实例，并加载预训练的权重。pretrained=True表示使用在COCO数据集上预训练的模型权重.
model.eval() #将模型设置为评估模式。在评估模式下，模型不会进行梯度计算和参数更新，适用于推理阶段.

# 如果你的模型是在GPU上训练的，确保模型也在GPU上进行推理
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#检查是否有可用的GPU，如果有，则将设备设置为GPU（cuda），否则使用CPU（cpu）.
model = model.to(device)#将模型移动到指定的设备上，以便在该设备上进行推理.

# 加载图像并进行预处理
def preprocess_image(image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])##创建一个图像转换序列，用于将输入图像转换为模型可以接受的格式.
    return transform(image).unsqueeze(0)  # 添加batch维度 unsqueeze(0)：在转换后的图像张量的最前面添加一个维度，使其成为形状为[1, C, H, W]的张量，其中C是通道数，H是高度，W是宽度.
'''
transform = torchvision.transforms.Compose([...])：创建一个图像转换序列，包含一个转换操作.

torchvision.transforms.ToTensor()：将PIL图像转换为PyTorch张量，范围从[0, 255]缩放到[0.0, 1.0].
PyTorch张量.
具体作用：
将图像的像素值从整数（范围为[0, 255]）转换为浮点数（范围为[0.0, 1.0]）.
将图像的通道顺序从PIL的RGB格式转换为PyTorch张量的CHW（通道、高度、宽度）格式.
深度学习模型通常使用浮点数进行计算，范围为[0.0, 1.0]的浮点数可以更好地适应模型的训练和推理过程.
return transform(image).unsqueeze(0)：对图像进行转换，并使用unsqueeze(0)在张量的最前面添加一个批次维度，因为模型需要一个批次的输入，即使只有一个图像.


PIL图像
数据类型：PIL图像的数据类型通常是整数（uint8），像素值范围为[0, 255]。
通道顺序：PIL图像的通道顺序是RGB（红、绿、蓝），即图像数据是按RGB顺序排列的。
用途：PIL库主要用于图像的加载、保存和基本的图像处理操作，如裁剪、旋转等。
PyTorch张量
数据类型：PyTorch张量的数据类型通常是浮点数（如float32），像素值范围为[0.0, 1.0]。
通道顺序：PyTorch张量的通道顺序是CHW（通道、高度、宽度），即图像数据的排列顺序是先通道，然后是高度和宽度。
用途：PyTorch张量是深度学习模型的输入和输出格式，支持高效的数值计算和梯度传播。
'''
# 进行推理
def infer(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    return prediction

# 显示结果
def show_result(image, predictions):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    instance_colors = {}
    for pred in predictions:
        masks = pred['masks'].cpu().numpy() #masks是从预测结果字典中提取的掩码信息.掩码用于表示每个实例在图像中的位置和形状.提取掩码，并将其从GPU（如果有的话）移动到CPU，并转换为NumPy数组.
        labels = pred['labels'].cpu().numpy()#提取标签，并将其从GPU（如果有的话）移动到CPU，并转换为NumPy数组. 标签表示每个实例的类别，例如人、汽车等.
        scores = pred['scores'].cpu().numpy()#提取置信度分数，并将其从GPU（如果有的话）移动到CPU，并转换为NumPy数组. 置信度分数表示模型对每个实例的预测置信度，通常用于过滤掉不准确的预测.
        for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
            if score > 0.5: #检查置信度分数是否大于0.5，如果是，则认为该实例是有效的.
                mask = mask[0] #取掩码的第一个通道（因为掩码是三维的，第一个维度是通道）                 '''在Mask R-CNN模型中，每个实例的掩码是一个三维张量，形状通常为[1, H, W]，其中H和W分别是图像的高度和宽度。mask[0]表示取这个三维张量的第一个通道，将其转换为二维张量[H, W]，这样可以更方便地进行后续处理。'''
                mask = (mask > 0.5).astype(np.uint8) #将掩码转换为二值掩码，值为0或1. #mask > 0.5：这是一个布尔操作，将掩码中的每个像素值与0.5进行比较，生成一个布尔数组，其中值大于0.5的像素为True，否则为False。
                # .astype(np.uint8)：将布尔数组转换为整数数组，True转换为1，False转换为0，数据类型为np.uint8（无符号8位整数）。
                # 二值掩码可以更清晰地表示实例的边界，值为1的像素表示该像素属于实例，值为0的像素表示不属于实例。 这种二值掩码在进行轮廓检测等操作时非常有用。
                if i not in instance_colors:
                    instance_colors[i] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))  #生成一个包含三个随机整数的元组，分别表示颜色的R（红）、G（绿）、B（蓝）通道。
                color = instance_colors[i] #获取该实例的颜色.
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #使用OpenCV的findContours函数检测掩码的轮廓.
                cv2.drawContours(image, contours, -1, color, 2) #将轮廓绘制在图像上，使用指定的颜色和线宽.

    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用示例
image_path = 'street.jpg'  # 替换为你的图像路径
prediction = infer(image_path) #infer函数进行推理，获取预测结果.
image = Image.open(image_path)
image = show_result(image, prediction)

'''cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)：这是OpenCV的轮廓检测函数。
mask：输入的二值掩码图像。
cv2.RETR_TREE：轮廓检索模式，表示检索所有轮廓，并重建嵌套轮廓的层次结构。
cv2.CHAIN_APPROX_SIMPLE：轮廓近似方法，表示去除所有冗余点，压缩轮廓，从而节省存储空间。
contours：检测到的轮廓列表，每个轮廓是一个点集。
_：返回的轮廓的层次结构，这里我们不需要，所以用_忽略。

cv2.drawContours(image, contours, -1, color, 2)：这是OpenCV的轮廓绘制函数。
image：输入的图像，将在该图像上绘制轮廓。
contours：检测到的轮廓列表。
-1：表示绘制所有轮廓。
color：绘制轮廓的颜色，是一个包含三个整数的元组，表示R、G、B通道。
2：轮廓的线宽。'''