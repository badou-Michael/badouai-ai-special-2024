import torch
import numpy as np
import cv2
from hw11_alexnet_train import AlexNet  # 从 hw11_alexnet_train 导入 AlexNet

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化模型并加载权重
    model = AlexNet(num_classes=2)
    model_path = r".\Course_CV\Week11\alexnet\best_model_torch.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 加载待预测的图像
    image_path = r".\Course_CV\Week11\alexnet\AlexNet-Keras-master\test_baozi1.jpg"
    # 使用 OpenCV 读取图像
    img = cv2.imread(image_path)
    # 调整图像大小为 227x227（AlexNet 输入尺寸）
    img_resized = cv2.resize(img, (227, 227))
    # 将图像从 BGR 转换为 RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # 将图像数据类型转换为 float32 并归一化到 [0,1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    # 调整维度，从 [H, W, C] 变为 [C, H, W]
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    # 转换为张量
    img_tensor = torch.from_numpy(img_transposed)
    # 增加批次维度
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    # 进行预测
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    # 输出结果
    classes = ['cat', 'dog']
    print(f"预测结果：{classes[predicted.item()]}")

    # 显示图像
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()