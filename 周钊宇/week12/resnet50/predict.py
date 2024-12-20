import keras
import cv2
from model.resnet import resnet50
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


model = resnet50()
model.summary()
img_path = "cat0.jpeg"
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
print(x.shape)
x = preprocess_input(x)


predict = model.predict(x)
print(predict.shape)
print("predicted:", decode_predictions(predict))