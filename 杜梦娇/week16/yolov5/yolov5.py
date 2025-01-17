import torch
import cv2

# YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 将模型设置为评估模式
model.eval()

# 准备输入图像
image = cv2.imread('street.jpg')

# 使用模型进行推理
results = model(image)
print(results)
# 打印检测结果
print(results.xyxy[0])  # 打印检测框的坐标和置信度
print(results.pandas().xyxy[0])  # 以 pandas DataFrame 的形式打印检测结果

# 可视化检测结果
results.render()  # 在图像上绘制检测框
for img in results.render():
    cv2.imshow('YOLOv5 Detection', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
