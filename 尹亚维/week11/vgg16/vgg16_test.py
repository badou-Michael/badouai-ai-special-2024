from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# 加载预训练的 VGG16 模型（ImageNet 权重）
model = VGG16(weights='imagenet')

# 图片路径
img_path = './test_data/dog.jpg'  # 替换为实际图片路径

# 加载图片并调整大小到 VGG16 的输入形状 (224, 224, 3)
img = load_img(img_path, target_size=(224, 224))

# 转换图片为数组
img_array = img_to_array(img)

# 增加批次维度，从 (224, 224, 3) 转为 (1, 224, 224, 3)
img_array = np.expand_dims(img_array, axis=0)

# 使用 VGG16 的预处理方法对图片进行预处理
img_array = preprocess_input(img_array)

# 使用模型进行预测
predictions = model.predict(img_array)

# 解释预测结果，返回 Top-5 预测
decoded_predictions = decode_predictions(predictions, top=5)

# 打印预测结果
print("Predicted Results (Top-5):")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
    print(f"{i+1}: {label} ({score:.2f})")
