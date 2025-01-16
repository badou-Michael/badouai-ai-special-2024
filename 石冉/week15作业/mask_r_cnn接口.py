import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import cv2

#加载预训练模型
model=maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
#设置推理的设备，如果在gpu上训练就在gpu上推理，否则在cpu
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model=model.to(device)



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
def show_result(image_path,predictions):
    image=cv2.imread(image_path)
    instance_colors={}
    for pred in predictions:
        masks=pred['masks'].cpu().numpy()
        labels=pred['labels'].cpu().numpy()
        scores=pred['scores'].cpu().numpy()
        for i, (mask,label,score) in enumerate(zip(masks,labels,scores)):
            #如果分数大于 0.5，则认为该实例是有效的
            if score > 0.5:
                #将掩码转换为二值图像，其中像素值大于 0.5 的为 1，其余为 0.
                mask=mask[0]
                mask=(mask > 0.5).astype(np.uint8)
                #如果该实例还没有分配颜色，则随机生成一个颜色并存储在 instance_colors 字典中
                if i not in instance_colors:
                    instance_colors[i]=(np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))
                color= instance_colors[i]
                #使用 OpenCV 的 findContours 函数找到掩码的轮廓
                contours,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                #使用 drawContours 函数在图像上绘制轮廓，颜色为该实例的颜色，线宽为 2
                cv2.drawContours(image,contours,-1,color,2)
    cv2.imshow('Result',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

#开始推理
image_path='week15_street.jpg'
prediction=infer(image_path)
image=show_result(image_path,prediction)
