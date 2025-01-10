import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn

# 加载图片
image_path = 'E:/practice/八斗/课程/八斗AI2024精品班/【15】图像分割/代码/maskrcnn简单版street.jpg'

model = maskrcnn_resnet50_fpn(pretrained = True)
model.eval()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 图像预处理
def preprocess_image(images):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0,1)
        ]
    )
    return transform(images).unsqueeze(0)

def inference(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)
    image_tensor.to(device)

    with torch.no_grad():
        results = model(image_tensor)
    return results

def get_color(i):
    all_color = {}
    if i not in all_color:
        all_color[i] = (np.random.randint(0,255),np.random.randint(0,255))
    color = all_color[i]
    return color

def show_result(results):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    for result in results:
        masks = result['masks'].cpu().numpy()
        labels = result['labels'].cpu().numpy()
        scores = result['scores'].cpu().numpy()
        for i,(mask,label,score) in enumerate(zip(masks,labels,scores)):
            if score>0.5:
                # 获取2D掩膜
                mask = (mask[0]>0.5).astype(np.uint8)
                color = get_color(i)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, color, 2)

    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    result = inference(image_path)
    image = show_result(result)
