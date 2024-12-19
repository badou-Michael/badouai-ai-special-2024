import numpy as np
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import \
    BatchNormalization, Conv2D, ReLU, Input, MaxPool2D, AveragePooling2D,\
    Concatenate, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing import image

def conv_bn(
    input_tensor,
    filters,
    rows,
    cols,
    strides=(1,1),
    padding="same",
    name=None,
    use_bias=False
):
    if name != None:
        conv_name = name+"_conv"
        bn_name = name+"_bn"
    else:
        conv_name = None
        bn_name = None
    x = Conv2D(filters, (rows, cols), strides=strides, padding=padding, 
               name=conv_name, use_bias=use_bias)(input_tensor)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = ReLU(name=name)(x)
    return x

def inceptionv3(input_shape=[299,299,3], class_num=1000):
    input_tensor = Input(input_shape)
    
    x = conv_bn(input_tensor, 32, 3, 3, 2, padding="valid")  # (149,149,32)
    x = conv_bn(x, 32, 3, 3, padding="valid") # (147,147,32)
    x = conv_bn(x, 64, 3, 3) # (147,147,64)
    x = MaxPool2D((3,3),(2,2))(x)   # (73, 73, 64)
    x = conv_bn(x, 80, 1, 1, padding="valid")    # (73, 73, 80)
    x = conv_bn(x, 192, 3, 3, padding="valid")   # (71, 71, 192)
    x = MaxPool2D((3,3), (2,2))(x)  #  (35, 35, 192)
    
    # block=1, module=1
    branch1 = conv_bn(x, 64, 1, 1)  # (35, 35, 64)
    
    branch2 = conv_bn(x, 48, 1, 1)  # (35, 35, 48)
    branch2 = conv_bn(branch2, 64, 5, 5)    # (35, 35, 64)
    
    branch3 = conv_bn(x, 64, 1, 1)  # (35, 35, 64)
    branch3 = conv_bn(branch3, 96, 3, 3)    # (35, 35, 96)
    branch3 = conv_bn(branch3, 96, 3, 3)    # (35, 35, 96)
    
    branch4 = AveragePooling2D((3,3),(1,1), padding="same")(x) # (35, 35, 192)
    branch4 = conv_bn(branch4, 32, 1, 1)    # (35, 35, 32)
    
    x = Concatenate(axis=-1, name="mixed01")(\
        [branch1, branch2, branch3, branch4])
    # output:(35, 35, 64+64+192+32=256)
    
    # block=1, module=2
    branch1 = conv_bn(x, 64, 1, 1)
    
    branch2 = conv_bn(x, 48, 1, 1)
    branch2 = conv_bn(branch2, 64, 5, 5)
    
    branch3 = conv_bn(x, 64, 1, 1)
    branch3 = conv_bn(branch3, 96, 3, 3)
    branch3 = conv_bn(branch3, 96, 3, 3)
    
    branch4 = AveragePooling2D((3,3), (1,1), padding="same")(x)
    branch4 = conv_bn(branch4, 64, 1, 1)
    
    x = Concatenate(axis=-1)(\
        [branch1, branch2, branch3, branch4])
    # output:(35,35,64*3+96=288)
    
    # block1, module3
    branch1 = conv_bn(x, 64, 1, 1)
    
    branch2 = conv_bn(x, 48, 1, 1)
    branch2 = conv_bn(branch2, 64, 5, 5)
    
    branch3 = conv_bn(x, 64, 1, 1)
    branch3 = conv_bn(branch3, 96, 3, 3)
    branch3 = conv_bn(branch3, 96, 3, 3)
    
    branch4 = AveragePooling2D((3,3), (1,1), padding="same")(x)
    branch4 = conv_bn(branch4, 64, 1, 1)
    
    x = Concatenate(axis=-1)(\
        [branch1, branch2, branch3, branch4])
    # output:(35,35,64*3+96=288)
    
    # block2, module2
    branch1 = conv_bn(x, 384, 3, 3, (2,2), padding="valid")  # (17,17,384)
    
    branch2 = conv_bn(x, 64, 1, 1)  # (35, 35, 64)
    branch2 = conv_bn(branch2, 96, 3, 3)    # (35, 35, 96)
    branch2 = conv_bn(branch2, 96, 3, 3, (2,2), padding="valid") # (17, 17, 96)
    
    branch3 = MaxPool2D((3,3),(2,2))(x) # (17, 17, 288)
    
    x = Concatenate(axis=-1)([branch1, branch2, branch3])
    # output:(17,17,384+96+288=768)
    
    # bloch2, module2
    branch1 = conv_bn(x, 192, 1, 1) # (17, 17, 192)
    
    branch2 = conv_bn(x, 128, 1, 1) # (17, 17, 128)
    branch2 = conv_bn(branch2, 128, 1, 7)   # (17, 17, 128)
    branch2 = conv_bn(branch2, 192, 7, 1)   # (17, 17, 192)
    
    branch3 = conv_bn(x, 128, 1, 1, )   # (17, 17, 128)
    branch3 = conv_bn(branch3, 128, 7, 1)   # (17, 17, 128)
    branch3 = conv_bn(branch3, 128, 1, 7)   # (17, 17, 128)
    branch3 = conv_bn(branch3, 128, 7, 1)   # (17, 17, 128)
    branch3 = conv_bn(branch3, 192, 1, 7)   # (17, 17, 192)
    
    branch4 = AveragePooling2D((3,3), (1,1), padding="same")(x) # (17, 17, 768)
    branch4 = conv_bn(branch4, 192, 1, 1)   # (17, 17, 192)
    
    x = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
    # output:(17, 17, 192+192+192+192=768)
    
    # block2, module3,4
    for i in range(2):
        branch1 = conv_bn(x, 192, 1, 1) # (17, 17, 192)
    
        branch2 = conv_bn(x, 160, 1, 1) # (17, 17, 160)
        branch2 = conv_bn(branch2, 160, 1, 7)   # (17, 17, 160)
        branch2 = conv_bn(branch2, 192, 7, 1)   # (17, 17, 192)
        
        branch3 = conv_bn(x, 160, 1, 1, )   # (17, 17, 160)
        branch3 = conv_bn(branch3, 160, 7, 1)   # (17, 17, 160)
        branch3 = conv_bn(branch3, 160, 1, 7)   # (17, 17, 160)
        branch3 = conv_bn(branch3, 160, 7, 1)   # (17, 17, 160)
        branch3 = conv_bn(branch3, 192, 1, 7)   # (17, 17, 192)
        
        branch4 = AveragePooling2D((3,3), (1,1), padding="same")(x) # (17, 17, 768)
        branch4 = conv_bn(branch4, 192, 1, 1)   # (17, 17, 192)
        
        x = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
        # output:(17, 17, 192+192+192+192=768)
    
    # block2, module5
    branch1 = conv_bn(x, 192, 1, 1) # (17, 17, 192)
    
    branch2 = conv_bn(x, 192, 1, 1) # (17, 17, 192)
    branch2 = conv_bn(branch2, 192, 1, 7)   # (17, 17, 192)
    branch2 = conv_bn(branch2, 192, 7, 1)   # (17, 17, 192)
    
    branch3 = conv_bn(x, 192, 1, 1, )   # (17, 17, 192)
    branch3 = conv_bn(branch3, 192, 7, 1)   # (17, 17, 192)
    branch3 = conv_bn(branch3, 192, 1, 7)   # (17, 17, 192)
    branch3 = conv_bn(branch3, 192, 7, 1)   # (17, 17, 192)
    branch3 = conv_bn(branch3, 192, 1, 7)   # (17, 17, 192)
    
    branch4 = AveragePooling2D((3,3), (1,1), padding="same")(x) # (17, 17, 768)
    branch4 = conv_bn(branch4, 192, 1, 1)   # (17, 17, 192)
    
    x = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
    # output:(17, 17, 192+192+192+192=768)
    
    # block3, module1
    branch1 = conv_bn(x, 192, 1, 1) # (17, 17, 192)
    branch1 = conv_bn(branch1, 320, 3, 3, (2,2), padding="valid")    # (8, 8, 320)
    
    branch2 = conv_bn(x, 192, 1, 1) # (17, 17, 192)
    branch2 = conv_bn(branch2, 192, 1, 7)   # (17, 17, 192)
    branch2 = conv_bn(branch2, 192, 7, 1)   # (17, 17, 192)
    branch2 = conv_bn(branch2, 192, 3, 3, (2,2), padding="valid")    # (8, 8, 192)
    
    branch3 = MaxPool2D((3,3),(2,2))(x) # (8,8,768)
    
    x = Concatenate(axis=-1)([branch1, branch2, branch3])
    # output:(8, 8, 1280)
    
    # block3, module2
    branch1 = conv_bn(x, 320, 1, 1) # (8, 8, 320)
    
    branch2 = conv_bn(x, 384, 1, 1) # (8, 8, 384)
    branch2_1 = conv_bn(branch2, 384, 1, 3) # (8, 8, 384)
    branch2_2 = conv_bn(branch2, 384, 3, 1) # (8, 8, 384)
    branch2_c = Concatenate(axis=-1)([branch2_1, branch2_2])    # (8, 8, 768)
    
    branch3 = conv_bn(x, 448, 1, 1) # (8, 8, 448)
    branch3 = conv_bn(branch3, 384, 3, 3)   # (8, 8, 384)
    branch3_1 = conv_bn(branch3, 384, 1, 3) # (8, 8, 384)
    branch3_2 = conv_bn(branch3, 384, 3, 1) # (8, 8, 384)
    branch3_c = Concatenate(axis=-1)([branch3_1, branch3_2])    # (8, 8, 768)
    
    branch4 = AveragePooling2D((3,3), (1,1), padding="same")(x) # (8, 8, 1280)
    branch4 = conv_bn(branch4, 192, 1, 1)   # (8, 8, 192)
    
    x = Concatenate(axis=-1)([branch1, branch2_c, branch3_c, branch4])
    # output:(8,8,320+768+1280+192=2048)
    
    # block3, module3
    branch1 = conv_bn(x, 320, 1, 1) # (8, 8, 320)
    
    branch2 = conv_bn(x, 384, 1, 1) # (8, 8, 384)
    branch2_1 = conv_bn(branch2, 384, 1, 3) # (8, 8, 384)
    branch2_2 = conv_bn(branch2, 384, 3, 1) # (8, 8, 384)
    branch2_c = Concatenate(axis=-1)([branch2_1, branch2_2])    # (8, 8, 768)
    
    branch3 = conv_bn(x, 448, 1, 1) # (8, 8, 448)
    branch3 = conv_bn(branch3, 384, 3, 3)   # (8, 8, 384)
    branch3_1 = conv_bn(branch3, 384, 1, 3) # (8, 8, 384)
    branch3_2 = conv_bn(branch3, 384, 3, 1) # (8, 8, 384)
    branch3_c = Concatenate(axis=-1)([branch3_1, branch3_2])    # (8, 8, 768)
    
    branch4 = AveragePooling2D((3,3), (1,1), padding="same")(x) # (8, 8, 2048)
    branch4 = conv_bn(branch4, 192, 1, 1)   # (8, 8, 192)
    
    x = Concatenate(axis=-1)([branch1, branch2_c, branch3_c, branch4])
    # output:(8,8,320+768+1280+192=2048)
    
    x = GlobalAveragePooling2D(name="average_pooling")(x)   # (batch_size, 2048)
    
    x = Dense(1000, activation="softmax")(x)    # (batch_size, 1000)
    
    model = Model(inputs=input_tensor, outputs=x)
    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    
    return model

def img_preprocess(image):
    image /= 255.0
    image -= 0.5
    image *= 2.0
    return image

if __name__ == "__main__":
    model = inceptionv3()
    
    img = image.load_img("elephant.jpg", target_size=(299,299))
    img = image.img_to_array(img)
    img = img_preprocess(img)
    img = np.expand_dims(img, 0)
    
    print("predicted:", decode_predictions(model.predict(img)))
    