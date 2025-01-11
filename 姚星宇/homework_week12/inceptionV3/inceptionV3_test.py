from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# [此处插入完整的InceptionV3模型定义代码]

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def load_and_predict(img_path, model_path):
    # 加载整个模型
    model = load_model(model_path)

    # 预处理图像
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # 预测
    preds = model.predict(x)
    predicted_class_index = (preds > 0.7).astype(int)
    class_labels = ['cat', 'dog']
    predicted_class_label = class_labels[predicted_class_index[0][0]]
    print(f"Predicted class: {predicted_class_label}")

if __name__ == '__main__':
    test_image_path = './cat_test.jpg'  
    model_path = './inception_v3_model.h5'  
    load_and_predict(test_image_path, model_path)