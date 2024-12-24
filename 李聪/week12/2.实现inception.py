from inceptionV3 import InceptionV3
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np

# 加载 InceptionV3 模型
model = InceptionV3(input_shape=(299, 299, 3), classes=1000)

# 加载预训练权重
model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

# 加载图片并进行预处理
img_path = 'elephant.jpg'  
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 使用模型进行预测
preds = model.predict(x)

# 输出预测结果
print('Predicted:', decode_predictions(preds, top=3))  # 输出 Top 3 预测
