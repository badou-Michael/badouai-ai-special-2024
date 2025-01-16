import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import numpy as np
import cv2

# 加载预训练模型
model = maskrcnn_resnet50_fpn(pretrained = True)
model.eval()

# 如果模型是在GPU上训练的，确保模型也在GPU上进行推理，否则在cpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
# 图像进行预处理
def preprocess_image(image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return  transform(image).unsqueeze(0)
# 推理
def infer(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction
def show_result(image, predictions):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    color_mapping = {
        1:(255, 0, 0), #人蓝色表示
        2:(0, 255, 0), #自行车绿色表示
        3:(0, 0, 255)  #汽车用红色表示
    }
    for pred in predictions:
        masks = pred['masks'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        for mask, label, score in zip(masks,labels,scores):
            if score > 0.5:
                mask = mask[0]
                mask = (mask > 0.5).astype(np.uint8)
                #这行代码从color_mapping字典中获取当前标签对应的颜色。
                # 如果标签不在字典中，则默认使用白色（255, 255, 255）。
                color = color_mapping.get(label.item(),(255,255,255))
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, color, 2)
    image = cv2.resize(image,(700,700))
    cv2.imshow('Result',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 使用示例
image_path = '../week15/maskrcnn_simple/street.jpg'
prediction = infer(image_path)
image = Image.open(image_path)
image = show_result(image,prediction)