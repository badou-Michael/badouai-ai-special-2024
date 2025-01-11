import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
from model.mobilenet import MobileNet

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
if __name__=="__main__":

    img = image.load_img("elephant.jpg", target_size=(224,224,3))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)


    print(img.shape)
    model = MobileNet()
    predicts = model.predict(img)
    print("predictions:", decode_predictions(predicts))

