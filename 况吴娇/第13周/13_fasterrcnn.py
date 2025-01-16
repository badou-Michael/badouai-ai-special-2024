import torch #导入PyTorch库，这是一个开源的机器学习库，广泛用于计算机视觉和自然语言处理。
import torchvision##torchvision库，它是PyTorch的一个扩展包，专门用于处理图像和视频。
from torchvision.models.detection import fasterrcnn_resnet50_fpn
#从torchvision的models.detection模块导入fasterrcnn_resnet50_fpn函数，这是一个用于目标检测的预训练模型，基于ResNet-50骨干网络和特征金字塔网络（FPN）。
from torchvision.transforms import functional as F
#导入torchvision.transforms.functional模块，并重命名为F。这个模块包含了一系列用于图像预处理的函数。
from PIL import Image,ImageDraw #从PIL（Python Imaging Library）库导入Image和ImageDraw类，用于图像的打开、处理和绘制。
import numpy as np

# 加载预训练模型
model = fasterrcnn_resnet50_fpn(pretrained=True)#这行代码调用fasterrcnn_resnet50_fpn函数来创建一个Faster R-CNN模型实例，并设置pretrained=True参数以加载预训练的权重。
#通常在加载预训练模型后，会将模型设置为评估模式，这可以通过调用model.eval()实现。这会关闭模型中的dropout和batch normalization层，使其在推理时表现一致。
model.eval() #将模型设置为评估模式，这会关闭模型中所有在训练时特有的行为，如Dropout。

# 如果你的模型是在GPU上训练的，确保模型也在GPU上进行推理
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #这行代码检查是否有可用的CUDA（GPU）设备，如果有，则device变量被设置为'cuda'，否则设置为'cpu'。
model = model.to(device) #将模型移动到之前定义的device上，如果device是'cuda'，模型将被移动到GPU上，否则在CPU上。

# 加载图像并进行预处理
def preprocess_image(image):
    transform = torchvision.transforms.Compose([  #使用 Compose 来将这些步骤串联起来，从而简化代码并使其更加模块化。
        torchvision.transforms.ToTensor(), ## 定义转换操作：将PIL图像转换为Tensor
    ])
    return transform(image).unsqueeze(0)  # 添加batch维度
##由于PyTorch模型通常期望批量的输入数据，即使只处理一张图片，也需要添加一个批量维度。
# unsqueeze(0) 方法在Tensor的第一个维度（即第0维）添加了一个大小为1的新维度，从而将形状从 [C, H, W] 变为 [1, C, H, W]，其中 C 是通道数，H 是高度，W 是宽度。

# 进行推理
def infer(image_path):
    image = Image.open(image_path).convert("RGB") #打开指定路径的图像文件，并将其转换为RGB格式。convert("RGB") 确保图像是以红色、绿色和蓝色三个颜色通道的RGB模式表示的。
    image_tensor = preprocess_image(image) #调用preprocess_image函数对图像进行预处理，得到一个适合模型输入的张量。
    image_tensor = image_tensor.to(device) #将预处理后的图像张量移动到之前定义的device上。

    with torch.no_grad():
        prediction = model(image_tensor)

    return prediction
'''
将预处理后的图像张量移动到之前定义的device上。
'''

# 显示结果
def show_result(image, prediction):
    boxes = prediction[0]['boxes'].cpu().numpy() #从预测结果中提取边界框，并将其移动到CPU上，然后转换为NumPy数组。
    labels = prediction[0]['labels'].cpu().numpy() #从预测结果中提取标签，并将其移动到CPU上，然后转换为NumPy数组。
    scores = prediction[0]['scores'].cpu().numpy() #从预测结果中提取得分，并将其移动到CPU上，然后转换为NumPy数组。
    draw = ImageDraw.Draw(image) #创建一个ImageDraw对象，用于在图像上绘制。
    #image = Image.fromarray(np.uint8(image))

    for box, label, score in zip(boxes, labels, scores):  #使用zip函数同时遍历边界框、标签和得分
        if score > 0.5:  # 阈值可根据需要调整 如果得分高于0.5，则认为检测结果可信。
            top_left = (box[0], box[1]) #提取边界框的左上角坐标。
            bottom_right = (box[2],box[3]) #提取边界框的右下角坐标。
            draw.rectangle([top_left, bottom_right], outline='red', width=2) #使用draw.rectangle方法在图像上绘制边界框。
            # width 参数是可选的。width 参数用于指定绘制形状（如矩形）的线宽。如果不提供 width 参数或者将其设置为默认值，绘制的形状将使用默认的线宽。
            '''
            draw：这是一个 ImageDraw 对象，它提供了在图像上绘制形状和文本的方法。
            top_left：一个元组，包含矩形左上角的坐标（x, y）。
            bottom_right：一个元组，包含矩形右下角的坐标（x, y）。
            outline：指定矩形边框的颜色，这里设置为 'red'，表示矩形的边框颜色为红色。
            width：指定边框的宽度，这里设置为 2，意味着矩形边框的宽度为2像素。
            
            text：这是 ImageDraw 对象的一个方法，用于在图像上绘制文本。
            (box[0], box[1] - 10)：一个元组，指定文本绘制的起始坐标。这里使用边界框的左上角坐标 (box[0], box[1]) 并向下偏移10像素，以避免文本与边界框重叠。
            str(label)：要绘制的文本内容，这里将标签 label 转换为字符串。
            fill='red'：指定文本的颜色，这里设置为红色。
            '''
            print(str(label))
            draw.text((box[0], box[1] - 10), str(label), fill='red')  #在边界框的左上角绘制标签文本。
    image.show()

# 使用示例
image_path = 'street.jpg'  # 替换为你的图像路径 定义一个变量image_path，存储要检测的图像的路径。
prediction = infer(image_path) #调用infer函数，传入图像路径，进行目标检测。
image = Image.open(image_path) #再次打开图像文件，用于显示结果。
image = show_result(image, prediction) #调用show_result函数，传入图像和预测结果，显示检测结果。
