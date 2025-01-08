import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image,ImageDraw

print(torch.__version__)
print(torchvision.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())


model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


# 加载图像并进行预处理
def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image


def detect_objects(image_path):
    image = preprocess_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
    return outputs

def draw_bounding_boxes(image, outputs):
    boxes = outputs[0]['boxes'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    draw = ImageDraw.Draw(image)

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            draw.text((x1, y1), f'{label}: {score:.2f}', fill='red')

    image.show()


if __name__ == '__main__':
    image_path = 'street.jpg'
    outputs = detect_objects(image_path)
    draw_bounding_boxes(Image.open(image_path), outputs)



