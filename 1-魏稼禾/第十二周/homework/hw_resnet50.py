import tensorflow
import numpy as np
from tensorflow.keras.layers import \
    Conv2D, ZeroPadding2D, BatchNormalization, ReLU, \
    MaxPool2D, AveragePooling2D, Flatten, Dense, Input, Add
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input

def Conv_Block(input_tensor, filters, kernel_size, stage, block, strides=(2,2)):
    conv_name = "res"+str(stage)+block+"_branch"
    bn_name = "bn"+str(stage)+block+"_branch"
    
    filter1, filter2, filter3 = filters
    
    x = Conv2D(filter1, (1,1), strides=strides, name=conv_name+"2a")(input_tensor)
    x = BatchNormalization(name=bn_name+"2a")(x)
    x = ReLU()(x)   # 和Activation("relu")等价，试验一下是不是
    
    x = Conv2D(filter2, kernel_size, padding="same",name=conv_name+"2b")(x)
    x = BatchNormalization(name=bn_name+"2b")(x)
    x = ReLU()(x)
    
    x = Conv2D(filter3, (1,1), name=conv_name+"2c")(x)
    x = BatchNormalization(name=bn_name+"2c")(x)
    
    short_cut = Conv2D(filter3, (1,1), strides=strides, name=conv_name+"1")(input_tensor)
    short_cut = BatchNormalization(name=bn_name+"1")(short_cut)
    
    x = Add()([x, short_cut])
    return ReLU()(x)
    
def Identi_Block(input_tensor, filters, kernel_size, stage, block):
    conv_name = "res"+str(stage)+block+"_branch"
    bn_name = "bn"+str(stage)+block+"_branch"
    
    filter1, filter2, filter3 = filters
    
    x = Conv2D(filter1, (1,1), name=conv_name+"2a")(input_tensor)
    x = BatchNormalization(name=bn_name+"2a")(x)
    x = ReLU()(x)
    
    x = Conv2D(filter2, kernel_size, padding="same", name=conv_name+"2b")(x)
    x = BatchNormalization(name=bn_name+"2b")(x)
    x = ReLU()(x)
    
    x = Conv2D(filter3, (1,1), name=conv_name+"2c")(x)
    x = BatchNormalization(name=bn_name+"2c")(x)
    
    x = Add()([x, input_tensor])
    return ReLU()(x)
    
def ResNet50(input_size=[224,224,3], class_num=1000):
    img_input = Input(shape=input_size)
    
    # stage 0
    x = ZeroPadding2D((3,3))(img_input)
    x = Conv2D(64,kernel_size=(7,7),strides=(2,2), name="conv1")(x) # (112,112,64)
    # 是不是和下面的代码等价？
    # x = Conv2D(64, kernel_size(7,7), strides(2,2), padding="same", name="conv1")(x)
    x = BatchNormalization(name="batch1")(x)
    x = ReLU()(x)
    x = MaxPool2D((3,3),strides=(2,2), name="maxpool1")(x)  # (55,55,64)
    
    # stage 2
    x = Conv_Block(x, (64, 64, 256), (3,3), 2, "a", strides=(1,1))  # (55,55,64)
    x = Identi_Block(x, (64, 64, 256), (3,3), 2, "b")
    x = Identi_Block(x, (64, 64, 256), (3,3), 2, "c")
    
    # stage 3
    x = Conv_Block(x, (128, 128, 512), (3,3), 3, "a")   # (28, 28, 512)
    x = Identi_Block(x, (128, 128, 512), (3,3), 3, "b")
    x = Identi_Block(x, (128, 128, 512), (3,3), 3, "c")
    x = Identi_Block(x, (128, 128, 512), (3,3), 3, "d")
    
    # stage 4
    x = Conv_Block(x, (256, 256, 1024), (3,3), 4, "a")  # (14, 14, 1024)
    x = Identi_Block(x, (256,256,1024),(3,3),4,"b")
    x = Identi_Block(x, (256,256,1024),(3,3),4,"c")
    x = Identi_Block(x, (256,256,1024),(3,3),4,"d")
    x = Identi_Block(x, (256,256,1024),(3,3),4,"e")
    x = Identi_Block(x, (256,256,1024),(3,3),4,"f")
    
    # stage 5
    x = Conv_Block(x, (512,512,2048), (3,3), 5, "a")    # (7, 7, 2048)
    x = Identi_Block(x, (512,512,2048),(3,3),5,"b")
    x = Identi_Block(x, (512,512,2048),(3,3),5,"c")
    
    # stage 6
    x = AveragePooling2D((7,7), name="avg_pool")(x) # (1,1,2048)
    x = Flatten()(x)    # (2048, )
    x = Dense(class_num, activation="softmax", name="fc1000")(x)
    
    model = Model(inputs=img_input, outputs=x, name="ResNet50")
    
    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    
    return model
    
if __name__ == "__main__":
    model = ResNet50()
    model.summary()
    img = image.load_img("elephant.jpg", target_size=(224,224))
    img_arr = image.img_to_array(img)
    # img_arr = np.array(img_arr)
    img_arr = np.expand_dims(img_arr, 0)
    img_arr = preprocess_input(img_arr) # 输入的归一化操作，[0,255]->[-1,1]
    
    pred = model.predict(img_arr)
    print("pred:", decode_predictions(pred,1))
    