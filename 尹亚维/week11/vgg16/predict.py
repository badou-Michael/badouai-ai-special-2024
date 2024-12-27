import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# 加载保存的模型
model = load_model('model/vgg16_model.h5')  # 使用 .h5 格式
# 如果保存为 SavedModel 格式，使用以下方式加载模型
# model = tf.keras.models.load_model('vgg16_saved_model')

# 图片路径
img_path = './test_data/dog.jpg'  # 替换为实际图片路径

# 加载图片并调整大小到 (224, 224, 3)
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img)

# 增加批次维度，从 (224, 224, 3) 变为 (1, 224, 224, 3)
img_array = np.expand_dims(img_array, axis=0)

# 归一化像素值到 [0, 1] 范围
img_array = img_array / 255.0

# 使用模型进行预测
logits = model.predict(img_array)  # 预测的 logits

# 最后结果进行 softmax 预测
softmax_predictions = tf.nn.softmax(logits).numpy()  # 转换为概率分布

# 获取预测的类别索引
predicted_class = np.argmax(softmax_predictions, axis=-1)

# 打印预测结果索引
print(f"Predicted class index: {predicted_class[0]}")

# # 打印预测概率
# print("Predicted probabilities:")
# print(softmax_predictions[0])


# 如果有类别映射文件 synset.txt，加载类别标签
def load_synset(synset_path):
    with open(synset_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


synset_path = 'synset.txt'  # 替换为实际的类别映射文件路径
class_labels = load_synset(synset_path)

# 打印类别标签
if predicted_class[0] < len(class_labels):
    print(f"Predicted class label: {class_labels[predicted_class[0]]}")
else:
    print("Class label not found in synset.")
