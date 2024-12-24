from model.inception_v3 import InceptionV3
import cv2
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
# from keras.applications.imagenet_utils import preprocess_input
import numpy as np

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__=="__main__":
    model = InceptionV3()
    model.summary()
    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    img = image.load_img("elephant.jpg", target_size=(299,299))
    img = image.img_to_array(img)
    
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    print(img.shape)

    
    predicts = model.predict(img)
    print("Prediction:", decode_predictions(predicts))


