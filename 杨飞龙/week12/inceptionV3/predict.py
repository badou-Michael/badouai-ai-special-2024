import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np

# 1. 加载InceptionV3模型并加载预训练权重
model = InceptionV3(weights='imagenet')


# 2. 预处理函数
def preprocess_image(img_path):
    # 加载图片并调整为299x299的尺寸，InceptionV3需要输入299x299大小的图像
    img = image.load_img(img_path, target_size=(299, 299))
    # 将图片转为数组
    img_array = image.img_to_array(img)
    # 增加批次维度
    img_array = np.expand_dims(img_array, axis=0)
    # 进行预处理，标准化输入数据（与训练时一致）
    img_array = preprocess_input(img_array)
    return img_array


# 3. 推理函数
def predict(img_path, model):
    # 预处理图像
    img_array = preprocess_image(img_path)

    # 使用模型进行预测
    preds = model.predict(img_array)

    # 解码预测结果，获取人类可读的标签
    decoded_preds = decode_predictions(preds, top=3)[0]  # top-3预测结果
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        print(f"{i + 1}: {label} ({score:.2f})")


# 测试推理
image_path = 'dog.jpg'  # 替换为您的图片路径
predict(image_path, model)
