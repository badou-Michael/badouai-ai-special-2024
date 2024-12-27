import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)


def preprocess_image(image):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    return transform(image).unsqueeze(0)


def infer(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    return prediction


def show_result(image, prediction):
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 30)
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='red', width=2)
            draw.text((box[0], box[1] - 30), str(label), fill='red', font=font)

    image.show()


if __name__ == '__main__':
    image_path = 'street.jpg'
    prediction = infer(image_path)
    image = Image.open(image_path)
    show_result(image, prediction)
