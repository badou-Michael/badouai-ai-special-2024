import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

def preprocess_image(image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def infer(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction

def show_result(image, prediction):
    boxs = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    draw = ImageDraw.Draw(image)

    for box, label, score in zip(boxs, labels, scores):
        if score > 0.5:
            top_left = (box[0], box[1])
            bottom_right = (box[2], box[3])
            draw.rectangle([top_left, bottom_right], outline='red', width=2)
            print(str(label))
            draw.text((box[0], box[1]-10), str(label), fill='red')
    image.show()

image_path = 'street.jpg'
prediction = infer(image_path)
image = Image.open(image_path)
image = show_result(image, prediction)