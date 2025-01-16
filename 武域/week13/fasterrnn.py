import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def process_image(image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def infer(image_path):
    image = Image.open(image_path).convert('RGB')
    image = process_image(image)
    image = image.to(device)
    with torch.no_grad():
        predictions = model(image)

    return predictions

def show_result(image, predictions):
    boxex = predictions[0]['boxes'].cpu().numpy()
    lables = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    draw = ImageDraw.Draw(image)

    for box, label, score in zip(boxex, lables, scores):
        if score > 0.5:
            tob_left = (box[0], box[1])
            bottom_right = (box[2], box[3])
            draw.rectangle([tob_left, bottom_right], outline='red')
            print(str(label))
            draw.text((box[0], box[1] - 10), str(label), fill='red')

    image.show()

image_path = 'street.jpg'
predictions = infer(image_path)
image = Image.open(image_path)
image = show_result(image, predictions)
