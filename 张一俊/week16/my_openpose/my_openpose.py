import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

# 加载预训练的OpenPose模型
model = torch.hub.load('CMU-Visual-Computing-Lab/openpose', 'pose_resnet50', pretrained=True)
model.eval()

# 图像预处理
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0)

# 读取图像
image_path = "demo.jpg"  # 替换为你的图片路径
image = cv2.imread(image_path)
image_tensor = preprocess_image(image)

# 推断和结果处理
with torch.no_grad():
    output = model(image_tensor)
# 表示关节点的热图
heatmaps = output[0].cpu().numpy()
keypoints = np.argmax(heatmaps, axis=0)
for i in range(keypoints.shape[0]):
    y, x = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

cv2.imshow('Keypoints', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
