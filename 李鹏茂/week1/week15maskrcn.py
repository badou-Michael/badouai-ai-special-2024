# import torch
# import torchvision
# from torchvision.models.detection import maskrcnn_resnet50_fpn
# from PIL import Image,ImageDraw
# import numpy as np
# import cv2
import  torch
import  torchvision
from torchvision.models.detection import  maskrcnn_resnet50_fpn
from  PIL import Image
import numpy as np
import cv2

# 加载预训练模型
# model = maskrcnn_resnet50_fpn(pretrained=True)
# model.eval()
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model.to(device)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model = model.to(device)


def press_image(img):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    return transform(img).unsqueeze(0)
def inference(img_path):
    img = Image.open(img_path).convert("RGB")
    img_ten = press_image(img)
    img_ten = img_ten.to(device)

    with torch.no_grad():
        predictions = model(img_ten)
    return predictions

def show_result(image, predictions):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    instance_colors = {}
    for pred in predictions:
        masks = pred['masks'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
            if score >0.5:
                mask = mask[0]
                mask = (mask >0.5).astype(np.uint8)
                if i not in instance_colors:
                    instance_colors[i] =(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                color = instance_colors[i]
                countour,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image,countour,0,color,2)

    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 使用示例
image_path = 'street.jpg'  # 替换为你的图像路径
prediction = inference(image_path)
image = Image.open(image_path)
image = show_result(image, prediction)
