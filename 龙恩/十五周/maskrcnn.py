import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image,ImageDraw
import numpy as np
import cv2


model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

def preprocess_image(image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  

def infer(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    return prediction

# show result
def show_result(image, predictions):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    instance_colors = {}
    for pred in predictions:
        masks = pred['masks'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
            if score > 0.5:
                mask = mask[0]
                mask = (mask > 0.5).astype(np.uint8)
                if i not in instance_colors:
                    instance_colors[i] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                color = instance_colors[i]
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, color, 2)

    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = 'street.jpg' 
prediction = infer(image_path)
image = Image.open(image_path)
image = show_result(image, prediction)

