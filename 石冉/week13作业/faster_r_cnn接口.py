import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image,ImageDraw
import numpy

#定义preprocess_image函数，将图片转换为tensor格式
def preprocess_image(image):
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    return transform(image).unsqueeze(0) #添加一个维度batch，从hwc变成nhwc

#定义infer函数，读取图片，转换格式，进行推理
def infer(image_path):
    image=Image.open(image_path).convert('RGB')
    image_tensor=preprocess_image(image)
    image_tensor=image_tensor.to(device)
    with torch.no_grad():
        prediction=model(image_tensor)
    return prediction

#定义显示结果函数，将结果画图显示出来
def show_result(image,prediciton):
    #从模型的输出中提取边界框信息，并将其从GPU（如果在使用）转移到CPU，然后转换为NumPy数组
    boxes=prediciton[0]['boxes'].cpu().numpy()
    labels=prediction[0]['labels'].cpu().numpy()
    scores=prediction[0]['scores'].cpu().numpy()
    draw=ImageDraw.Draw(image)

    #使用 zip 函数同时遍历边界框、标签和分数。
    #zip 是 Python 中的一个内置函数，它用于将多个可迭代对象（如列表、元组、字符串等）的对应元素打包成一个个元组，
    # 然后返回由这些元组组成的对象（通常是一个迭代器）。这个对象可以被用来在循环中同时迭代多个可迭代对象。
    for box,label,score in zip(boxes,labels,scores):
        #只处理置信度在0.5以上的，可以根据实际情况调整0.5
        if score>0.5:
            top_left=(box[0],box[1]) #左上角坐标
            bottom_right=(box[2],box[3]) #右下角坐标
            draw.rectangle([top_left,bottom_right],outline='red',width=2)#画出边框
            print(str(label)) #打印标签
            draw.text((box[0],box[1]-10),str(label),fill='red')# 在边界框的左上角上方绘制文本标签。
    image.show()

#加载预训练模型
model=fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
#设置推理的设备，如果在gpu上训练就在gpu上推理，否则在cpu
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model=model.to(device)

#开始推理
image_path='week13_street.jpg'
prediction=infer(image_path)
image=Image.open(image_path)
image=show_result(image,prediction)
