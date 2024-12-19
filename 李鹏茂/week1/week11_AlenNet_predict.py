import numpy as np
import utils
import cv2
from keras import backend as K
from model.AlexNet import AlexNet

# 设置图像数据格式，确保使用正确的格式
K.image_data_format() == 'channels_first'  # 确保使用的是 'channels_first' 还是 'channels_last'

if __name__ == "__main__":
    # 初始化AlexNet模型
    model = AlexNet()

    # 加载已训练的模型权重
    model.load_weights("./logs/last1.h5")

    # 读取待预测的图像
    img_path = "./test2.jpg"
    img = cv2.imread(img_path)

    # 将BGR图像转换为RGB格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 图像归一化处理，将像素值从0~255映射到0~1
    img_normalized = img_rgb / 255.0

    # 扩展图像维度以适应模型输入（batch_size, height, width, channels）
    img_normalized = np.expand_dims(img_normalized, axis=0)

    # 调整图像大小为224x224，适应AlexNet的输入要求
    img_resized = utils.resize_image(img_normalized, (224, 224))

    # 预测图像所属的类别
    predicted_class = np.argmax(model.predict(img_resized))

    # 打印预测结果
    print('The predicted class is: ', utils.print_answer(predicted_class))

    # 显示原始图像
    cv2.imshow("Input Image", img)

    # 等待键盘事件后关闭显示窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
