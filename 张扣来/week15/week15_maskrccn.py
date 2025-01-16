import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import numpy as np
import cv2

# 加载预训练模型,调用了maskrcnn_resnet50_fpn函数来创建一个Mask R-CNN模型实例
#pretrained = True参数表示加载预训练的权重。
# 这意味着模型的参数是已经在某个大型数据集（如COCO数据集）上训练好的
# 可以直接用于进行目标检测和分割任务，而无需从头开始训练。
model = maskrcnn_resnet50_fpn(pretrained = True)
'''
将模型设置为评估模式。
在评估模式下，模型的一些特定层（如Dropout层和BatchNorm层）的行为会与训练模式不同。
例如，Dropout层在训练模式下会随机丢弃一些神经元的输出以防止过拟合，而在评估模式下则不会丢弃任何输出，
以确保模型的输出是稳定的。BatchNorm层在训练模式下会计算每个批次的均值和方差，
而在评估模式下会使用训练过程中计算得到的全局均值和方差。
'''
model.eval()
'''
torch.cuda.is_available()检查当前系统是否支持CUDA（即是否有可用的NVIDIA GPU）。
如果支持CUDA，则device被设置为torch.device('cuda')，表示模型将在GPU上运行；
如果不支持CUDA，则device被设置为torch.device('cpu')，表示模型将在CPU上运行。
'''
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# 加载图像进行预处理
'''
torchvision.transforms.Compose函数可以将多个图像转换操作组合在一起，按照顺序依次对图像进行处理。
torchvision.transforms.ToTensor()
这是转换管道中的一个转换操作，它将输入的图像从PIL图像或NumPy数组格式转换为PyTorch张量（Tensor）格式。
在PyTorch中，图像数据通常以张量的形式进行处理，张量是一个多维数组，类似于NumPy数组，但可以在GPU上进行加速计算。
ToTensor转换操作会将图像的像素值从[0, 255]的整数范围缩放到[0.0, 1.0]的浮点数范围，并将图像的维度从(H, W, C)
（高度、宽度、通道数）转换为(C, H, W)，以符合PyTorch模型的输入要求。
'''
def preprocess_image(image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    #这行代码首先使用定义好的转换管道transform对输入的图像image进行转换，得到一个张量。
    #调用.unsqueeze(0)方法在张量的第0维（即最前面的维度）添加一个额外的维度。
    # 这是因为PyTorch模型通常期望输入数据是一个批次（batch）的形式，即使我们只输入一张图像，也需要将其视为一个包含单个图像的批次。
    return transform(image).unsqueeze(0)
# 推理
def infer(image_path):
    #这行代码使用PIL库中的Image.open函数打开指定路径的图像文件。
    #.convert('RGB')方法将图像转换为RGB颜色模式，确保图像有三个颜色通道（红、绿、蓝）。
    #这一步是为了确保图像格式与模型的输入要求一致，因为有些图像可能不是RGB格式（例如灰度图像）。
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess_image(image)
    #预处理后的图像张量image_tensor转移到之前定义的设备device上
    image_tensor = image_tensor.to(device)
    # PyTorch会停止自动计算梯度
    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction
# 显示结果
def show_result(image, predictions):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    instance_colors = {}
    for pred in predictions:
        #从预测结果pred中提取分割掩码，并将其从GPU（如果有的话）转移到CPU，然后转换为NumPy数组。
        masks = pred['masks'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        #遍历每个实例的掩码、标签和置信度分数。
        for i,(mask, label, score) in enumerate(zip(masks, labels, scores)):
            #只有置信度分数大于0.5的实例才会被可视化
            if score > 0.5:
                mask = mask[0]
                #将掩码中的值二值化，将大于0.5的值设置为1，其余值设置为0，并将数据类型转换为np.uint8。
                # 这一步是为了将掩码转换为二值图像，便于后续的轮廓检测
                mask = (mask > 0.5).astype(np.uint8)
                #检查当前实例的索引i是否已经分配了颜色。如果没有分配颜色，则生成一个随机颜色。
                if i not in instance_colors:
                    instance_colors[i] = (np.random.randint(0,256), np.random.randint(0,256),
                                          np.random.randint(0,256))
                    #instance_colors字典中获取当前实例的颜色
                    color = instance_colors[i]
                    #这行代码使用OpenCV的cv2.findContours函数检测掩码中的轮廓。
                    # cv2.RETR_TREE表示检索所有轮廓并重建嵌套轮廓的层次结构
                    # cv2.CHAIN_APPROX_SIMPLE表示压缩轮廓，删除所有冗余点，节省存储空间。
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    #这行代码使用OpenCV的cv2.drawContours函数将检测到的轮廓绘制到原始图像上。
                    # -1表示绘制所有轮廓，color是轮廓的颜色，2是轮廓的线宽。
                    cv2.drawContours(image, contours, -1, color, 2)
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 使用实例
image_path = '../week15/maskrcnn_simple/street.jpg'
prediction = infer(image_path)
image = Image.open(image_path)
image = show_result(image, prediction)