
'''
torch是PyTorch的核心库，提供了多维数组（张量）的支持、自动求导功能以及神经网络构建所需的工具
torchvision是PyTorch的一个扩展包，它包含了一些常用的视觉模型架构、数据集加载器以及图像转换工具。
fasterrcnn_resnet50_fpn是一个预训练的Faster R-CNN模型，它使用ResNet-50作为骨干网络，并结合了特征金字塔网络（FPN）来提取特征
torchvision.transforms.functional模块提供了一系列图像处理函数，这些函数可以直接应用于PIL图像或PyTorch张量。例如调整大小、裁剪、旋转、归一化等。
PIL（Python Imaging Library）提供了许多强大的图像处理功能。Image模块用于打开、操作和保存多种图像格式，而ImageDraw模块则用于在图像上绘制文本、线条、形状等
'''
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import numpy as np


'''
加载与训练模型，pretrained=True参数表示加载预训练的权重。这些权重是在COCO数据集上预训练得到的，因此模型已经具备了一定的目标检测能力。
eval()方法将模型设置为评估模式。在评估模式下，模型的行为与训练模式不同：
关闭Dropout层：在训练过程中，Dropout层会随机丢弃一些神经元的输出，以防止过拟合。但在评估模式下，所有神经元的输出都会被保留。
使用全局统计信息进行BatchNorm层的归一化：在训练过程中，BatchNorm层会使用每批次数据的统计信息进行归一化。而在评估模式下，
它会使用在训练过程中计算得到的全局统计信息。
将模型设置为评估模式非常重要，因为某些层在训练和评估时的行为不同，如果在评估时不切换模式，可能会导致不准确的预测结果。
'''
model = fasterrcnn_resnet50_fpn(pretrained = True)
model.eval()

# 检查是否有可用的CUDA设备（GPU），如果有则创建一个指向CUDA设备的device对象，
# 否则创建一个指向CPU的device对象。
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device) #将模型移动到指定的设备上（GPU或CPU）。

# 加载图像并进行预处理
def preprocess_image(image):
    '''
    torchvision.transforms.Compose：这是一个将多个图像变换组合在一起的工具。
    在这个例子中，它只包含一个变换，可以根据需要添加更多的变换，比如归一化、裁剪、旋转等。
    torchvision.transforms.ToTensor()：这个变换将PIL图像或NumPy ndarray转换为PyTorch的FloatTensor，
    并且将图像的像素值从[0, 255]缩放到[0.0, 1.0]。这是因为大多数深度学习模型期望输入的像素值在0到1之间。
    '''
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    '''
    unsqueeze(0)：这是一个PyTorch张量操作，它在指定的维度上增加一个维度。
    在这里，它在第0维增加了一个维度，使得单个图像的张量形状从 (C, H, W) 变为 (1, C, H, W)，其中 C 是通道数，H 是高度，W 是宽度。
    '''
    return transform(image).unsqueeze(0)
#进行推理
def infer(image_path):
    # 使用PIL库打开图像文件，并将其转换为RGB格式
    # 这样做是为了确保图像有三个颜色通道，即使原始图像是灰度图
    image = Image.open(image_path).convert('RGB')
    # print('image的数据内容：\n',image.size)
    image_tensor = preprocess_image(image)
    # 将预处理后的图像张量移动到之前定义的设备上（GPU或CPU）
    image_tensor = image_tensor.to(device)
    # 使用torch.no_grad()上下文管理器来禁用梯度计算
    # 在推理阶段，不需要计算梯度，这样做可以减少内存消耗并提高推理速度
    with torch.no_grad():
        # 使用模型对图像张量进行预测
        prediction = model(image_tensor)
    return prediction

# 在输入图像上绘制模型预测的结果
def show_result(image, prediction):
    # 从模型的预测结果中提取边界框、标签和置信度分数
    # 这里假设预测结果是一个列表，其中第一个元素包含了所有相关信息
    '''
    从预测结果中提取边界框坐标，并将其从PyTorch张量转换为NumPy数组。
    .cpu() 是将张量从GPU（如果使用了GPU）移动到CPU，因为Pillow库不能直接处理GPU上的张量。
    '''
    boxes = prediction[0]['boxes'].cpu().numpy() # 边界框坐标
    labels = prediction[0]['labels'].cpu().numpy() # 对应的标签
    scores = prediction[0]['scores'].cpu().numpy() # 置信度分数
    # 创建一个可以在图像上绘制的ImageDraw对象
    draw = ImageDraw.Draw(image)
    # 遍历每一个预测的边界框、标签和分数
    for box,labels,score in zip(boxes, labels, scores):
        # 只有当置信度分数大于0.5时，才绘制边界框和标签
        if score > 0.5:
            # 定义边界框的左上角和右下角坐标
            top_left = (box[0], box[1])
            bottom_right = (box[2],box[3])
            # 在图像上绘制边界框，使用红色线条，宽度为2
            draw.rectangle([top_left, bottom_right], outline = 'red', width = 10)
            # 打印标签（在实际应用中可能需要将标签ID转换为实际的类别名称）
            print(str(labels))
            # 在边界框的左上角上方绘制标签文本，使用红色填充
            draw.text((box[0],box[1]- 10), str(labels), fill = 'red')
    image.show()
# 数据参数输入
image_path = '../task/week13/fasterrcnn_simple/street.jpg'
# 推理结果
prediction = infer(image_path)
# 这是Pillow库提供的一个函数，用于打开一个图像文件。它接受一个文件路径作为参数，并返回一个PIL图像对象。
# 这个对象包含了图像的各种信息，如大小、模式（颜色空间）、像素数据等
image = Image.open(image_path)
# 图像输入后预测结果
image = show_result(image, prediction)

