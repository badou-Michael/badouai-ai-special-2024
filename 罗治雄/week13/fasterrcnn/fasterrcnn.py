import torch
import torchvision
from collections import OrderedDict
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image,ImageDraw
import numpy as np


model = fasterrcnn_resnet50_fpn(pretrained=True)
# model.eval()是PyTorch中的一个关键方法，用于将模型设置为评估模式。
# 这种模式对于模型推理和验证至关重要，因为它确保模型在预测新数据时能够给出准确的结果
model.eval()

# torch.device('cuda')：这是一个指定设备为CUDA的PyTorch设备对象。
# torch.cuda.is_available()：这是一个函数，用于检查CUDA是否可用。如果可用，它返回True；如果不可用，它返回False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# to(device):将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行
model = model.to(device)

# transforms.Compose 是PyTorch库中torchvision.transforms模块提供的一个功能，它允许将多个图像变换操作组合起来。
# 当你在处理图像，并需要依次应用多个变换（如缩放、裁剪、归一化等）时，Compose可以把这些变换串联成一个单一的操作，
# 这样你就可以非常方便地在数据集上应用这个组合操作

def preprocess_image(image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  #添加batch维度

def infer(image_path):
#     使用Image.open读出图像 
# 读出来的图像是RGBA四通道的，A通道为透明通道，该通道值对深度学习模型训练来说暂时用不到，因此使用convert(‘RGB’)进行通道转换
    image = Image.open(image_path).convert('RGB')
# preprocess_image通常是一个预处理图像数据的函数，在深度学习或计算机视觉项目中非常常见，它的作用是将原始图片转换成模型可以
# 接受的标准输入格式。这个函数可能会执行一系列操作，如调整尺寸、归一化像素值、色彩空间转换等。
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)
#     torch.no_grad() 是 PyTorch 中的一个上下文管理器，用于在进入该上下文时禁用梯度计算。这在你只关心评估模型，而不是训练模型时非常有用，因为它可以显著减少内存使用并加速计算。

# 当你在 torch.no_grad() 上下文管理器中执行张量操作时，PyTorch 不会为这些操作计算梯度。
# 这意味着不会在 .grad 属性中累积梯度，并且操作会更快地执行。
    
    with torch.no_grad():
        prediction = model(image_tensor)
        
    return prediction

def show_result(image,prediction):
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    draw = ImageDraw.Draw(image)
    
    for box,label,score in zip(boxes,labels,scores):
        if score > 0.5:
            top_left = (box[0],box[1])
            bottom_right = (box[2],box[3])
            draw.rectangle([top_left,bottom_right],outline='red',width=2)
            print(str(label))
            draw.text((box[0],box[1] - 10),str(label),fill='red')
    image.show()
    

image_path = 'C://Users/luozx/week13/fasterrcnn简单版/street.jpg'
prediction = infer(image_path)
image = Image.open(image_path)
image = show_result(image,prediction)
