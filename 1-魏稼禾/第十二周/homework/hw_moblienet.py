import numpy as np
from tensorflow.keras.layers import \
    Input, Conv2D, DepthwiseConv2D, BatchNormalization, GlobalAveragePooling2D,\
    Activation, Dropout, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as k
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing import image
    
def net_body(input_shape=(224,224,3), 
                output_class=1000,
                dropout = 1e-3):
    inputs = Input(shape=input_shape)
    
    x = _conv_bn(inputs, 32, (3,3), (2,2)) # (112, 112, 32)
    
    x = _depthwise_conv(x, 64, depth_multiplier=1, layer_id=1) # (112, 112, 64)
    
    x = _depthwise_conv(x, 128, strides=(2,2), depth_multiplier=1, layer_id=2) #(56, 56, 128)
    
    x = _depthwise_conv(x, 128, depth_multiplier=1, layer_id=3)    # (56, 56, 128)
    
    x = _depthwise_conv(x, 256, strides=(2,2), depth_multiplier=1, layer_id=4) #(28, 28, 256)
    
    x = _depthwise_conv(x, 256, depth_multiplier=1, layer_id=5)   #(28, 28, 256)
    
    x = _depthwise_conv(x, 512, strides=(2,2), depth_multiplier=1, layer_id=6) #(14, 14, 512)
    
    for i in range(5):
        x = _depthwise_conv(x, 512, depth_multiplier=1, layer_id=i+7)  #(14, 14, 512)
        
    x = _depthwise_conv(x, 1024, strides=(2,2), depth_multiplier=1, layer_id=12)   #(7,7,1024)
    
    x = _depthwise_conv(x, 1024, depth_multiplier=1, layer_id=13)  #(7,7,1024)
    
    x = GlobalAveragePooling2D()(x) # (1024)
    x = Reshape((1,1,1024), name="reshape1")(x)  # (1,1,1024)
    x = Dropout(dropout, name="dropout")(x)
    x = Conv2D(output_class, (1,1), name="conv_14")(x)  # (batch, 1000)
    x = Reshape((output_class,), name="reshape2")(x)
    x = Activation("softmax", name="softmax")(x)    # (1000, )
    
    model = Model(inputs=inputs, outputs=x)
    
    model.load_weights("mobilenet_1_0_224_tf.h5")
    
    return model
    

def _conv_bn(input_tensor,
             filters,
             kernel_size = (3,3),
             strides = (1,1),
             padding="same",
             ):
    x = Conv2D(filters, kernel_size, strides, padding, use_bias=False, name="conv1")(input_tensor)
    x = BatchNormalization(name="bn1")(x)
    x = relu6(x)
    return x
    
def _depthwise_conv(input_tensor,
                    filters,
                    kernel_size=(3,3),
                    strides= (1,1),
                    padding="same",
                    depth_multiplier=1,
                    layer_id = 1
                    ):
    x = DepthwiseConv2D(
        kernel_size, strides, padding, depth_multiplier, use_bias=False, name="dw_conv_"+str(layer_id)
        )(input_tensor)
    x = BatchNormalization(name="dw_bn_"+str(layer_id))(x)
    x = relu6(x)
    x = Conv2D(filters, (1,1), use_bias=False, name="pw_conv_"+str(layer_id))(x)
    x = BatchNormalization(name="pw_bn_"+str(layer_id))(x)
    x = relu6(x)
    return x
        
def relu6(x):
        return k.relu(x, max_value=6)
    
def preprocess_img(img):
    img /= 255.0
    img -= 0.5
    img *= 2
    return img

if __name__ == "__main__":
    img = image.load_img("elephant.jpg", target_size=(224,224))
    img = image.img_to_array(img)
    img = preprocess_img(img)
    img = np.expand_dims(img, axis=0)
    model = net_body()
    
    print("预测为：", decode_predictions(model.predict(img)))