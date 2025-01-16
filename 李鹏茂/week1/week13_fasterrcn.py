import torch
import torchvision
from torchvision.models.detection  import  fasterrcnn_resnet50_fpn
from PIL import  Image, ImageDraw, ImageFont


modle =fasterrcnn_resnet50_fpn(pretrained=True)
modle.eval()

device =torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")
modle.to(device)
def proessimage(image):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    return transform(image).unsqueeze(0)
def infer (path):
    img = Image.open(path).convert("RGB")
    imgtensor= proessimage(img).to(device)
    with torch.no_grad():
        preds = modle(imgtensor)
    return  preds
# def show(img,preds):
#
#     bbox = prediction[0]['boxes'].cpu().numpy()
#     label = prediction[0]['labels'].cpu().numpy()
#     score = prediction[0]['scores'].cpu().numpy()
#     draw = ImageDraw.Draw(img)
#     for box,lab,sco in zip(bbox,label,score):
#         if score > 0.6:
#             tp_left = int(box[0]*img)
#             tp_right = int(box[1]*img)
#             tp_top = int(box[2]*img)
#             tp_bottom = int(box[3]*img)
#             draw.rectangle(([tp_left, tp_right],[tp_top, tp_bottom]), outline='red', width=2)
#             draw.text((box[0], box[1] - 10), str(label), fill='red')
#     img.show()
#
#
# image_path = 'street.jpg'  # 替换为你的图像路径
# prediction = infer(image_path)
# image = Image.open(image_path)
# image = show(image, prediction)
def show_result(image, prediction):
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    draw = ImageDraw.Draw(image)
    # image = Image.fromarray(np.uint8(image))

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # 阈值可根据需要调整
            top_left = (box[0], box[1])
            bottom_right = (box[2], box[3])
            draw.rectangle([top_left, bottom_right], outline='red', width=2)
            print(str(label))
            draw.text((box[0], box[1] - 10), str(label), fill='red')
    image.show()


# 使用示例
image_path = 'street.jpg'  # 替换为你的图像路径
prediction = infer(image_path)
image = Image.open(image_path)
image = show_result(image, prediction)

