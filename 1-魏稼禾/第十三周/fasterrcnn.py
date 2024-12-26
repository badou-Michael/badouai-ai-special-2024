import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image, ImageDraw

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

# COCO 数据集的类别标签
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

def img_preprocess(img):
    process = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    return process(img).unsqueeze(0)    #增加batch_size层

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = img_preprocess(img)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        prediction = model(img_tensor)
    return prediction, img
    
def show_prediction(image, prediction):
    draw = ImageDraw.Draw(image)
    """
    prediction = [{
        "boxes" = tensor([[x_min, y_min, x_max, y_max], ...]),
        "labels" = tensor([label1, label2, ...])
        "scores" = tensor([score1, score2, ...])
    }]
    """
    scores = prediction[0]["scores"].cpu().detach().tolist()
    labels = prediction[0]["labels"].cpu().detach().tolist()
    boxes = prediction[0]["boxes"].cpu().detach().tolist()
    
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            top_left = (box[0], box[1])
            bottom_right = (box[2], box[3])
            draw.rectangle((top_left, bottom_right), outline="red", width=2)
            draw.text((box[0],box[1]-10), "%.3f %s"%(score,COCO_INSTANCE_CATEGORY_NAMES[label]), fill="red")
    image.show()

img_path = "street.jpg"
prediction, img = predict(img_path)
show_prediction(img, prediction)