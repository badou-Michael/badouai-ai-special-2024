# 引入需要的包
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import PIL.ImageDraw as ImageDraw
from PIL.Image import  fromarray
from PIL import Image
import numpy as np
import cv2
import PIL.ImageFont as ImageFont
from PIL import ImageColor

# 加载图像并进行预处理
def preprocess_image(image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # 添加batch维度

# 模型推理
def infer_self(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = model(image_tensor)
    # print("prediction = ",prediction)
    return prediction


# 显示结果
def draw(image_path, predictions,box_thresh,mask_thresh):
    image_data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    visualization = img.copy()
    instance_colors = []
    for pred in predictions:
        masks = pred['masks'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
            if score > box_thresh:
                mask = mask[0]
                mask = (mask > mask_thresh).astype(np.uint8)
                instance_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                while instance_color in instance_colors:
                    instance_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(visualization, contours, -1, instance_color, 2)
                for contour in contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        
                        cv2.putText(visualization, str(label), [contour[0][0][0],contour[0][0][1]], cv2.FONT_HERSHEY_SIMPLEX,2,instance_color,2)

    cv2.imshow('Result', visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
        


# 加载预训练模型
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
# 将模型放到设备上
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# 读取一张图片
image_path = 'E:\学习文件\第15周\homework_for_week_15\mask_rcnn\street.jpg'
prediction = infer_self(image_path)

box_thresh = 0.7
mask_thresh = 0.7

draw(image_path, prediction,box_thresh,mask_thresh)
