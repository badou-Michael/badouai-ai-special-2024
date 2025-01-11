import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision
import torch
from PIL import ImageDraw,Image

def proccessInput(imgarray):
    """
    将ndarray转为tensor,并增加一个维度
    @param imgarray:
    @return:
    """
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    return transforms(imgarray).unsqueeze(0)  #添加batch维度

def showresult(image,predict):
    predict_dict = predict[0]
    boxes_tensor = predict_dict["boxes"]
    boxes_tensor_numpy = boxes_tensor.cpu().numpy()
    labels_tensor = predict_dict["labels"]
    labels_tensor_numpy = labels_tensor.cpu().numpy()
    scores_tensor = predict_dict["scores"]
    scores_tensor_numpy = scores_tensor.cpu().numpy()
    # draw = ImageDraw.Draw(image)
    for box,label,score in zip(boxes_tensor_numpy,labels_tensor_numpy,scores_tensor_numpy):
        if score>0.7:
            top_left =(box[0],box[1])
            bottom_right = (box[2],box[3])
            # draw.rectangle(xy=[top_left,bottom_right],outline="red",width=1)
            # draw.text(xy=[box[0],box[1]-10],text=str(label),fill="red")
            cv2.rectangle(image,top_left,bottom_right,color=(0,0,255))
            cv2.putText(image, str(label), (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255), 1)
    return image

if __name__ == '__main__':
    impath="test2.jpg"
    oriimg = cv2.imread(impath)
    img = cv2.cvtColor(oriimg,cv2.COLOR_BGR2RGB)
    #数据预处理
    image_tensor = proccessInput(img)

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # 设置模型为评估模式
    model.eval()
    # 如果你的模型是在GPU上训练的，确保模型也在GPU上进行推理
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        prediction = model(image_tensor)
    image_open = Image.open(impath)
    resultimg = showresult(oriimg, prediction)
    # image_open.show()
    cv2.imshow("resultimg",resultimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()