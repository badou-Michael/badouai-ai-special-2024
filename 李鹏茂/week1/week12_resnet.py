# lpm 2024/12/19
from keras import layers
import numpy as np
from keras.preprocessing.image import image
import  keras.backend as K
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import  decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model

def id_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_bas = 'res' + str(stage) + block+'_branch'
    bn_name_bas = 'bn'+ str(stage)+block + '_branch'
    x = layers.Conv2D(filters1,(1,1),name=conv_name_bas+'2a')(input_tensor)
    x = layers.BatchNormalization(name=bn_name_bas+ '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2,kernel_size,padding='same',name=conv_name_bas+'2b')(x)
    x = layers.BatchNormalization(name=bn_name_bas+ '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3,(1,1),name=conv_name_bas+'2c')(x)
    x = layers.BatchNormalization(name=bn_name_bas+ '2c')(x)
    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)

    return  x
def conv_block(input_tensor, kernel_size, filters, stage, block,strides=(2,2)):
    filters1, filters2, filters3 = filters

    conv_name_bas = 'res' + str(stage) + block + '_branch'
    bn_name_bas = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),strides=strides, name=conv_name_bas +'2a')(input_tensor)
    x = layers.BatchNormalization(name=bn_name_bas + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',name=conv_name_bas + '2b')(x)
    x = layers.BatchNormalization(name=bn_name_bas + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1,1), name=conv_name_bas + '2c')(x)
    x = layers.BatchNormalization(name=bn_name_bas + '2c')(x)


    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,name=conv_name_bas +'1')(input_tensor)
    shortcut = layers.BatchNormalization(name=bn_name_bas+'1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return  x
def ResNet50(input_tensor_shape=[224,224,3], num_classes=1000):
    image = layers.Input(shape=input_tensor_shape)
    x=  layers.ZeroPadding2D((3,3))(image)
    x = layers.Conv2D(64, kernel_size=(7,7), strides=(2,2),name = 'con1')(x)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3,3), strides=(2,2))(x)



    x = conv_block(x,3,[64,64,256],stage=2,block ='a',strides=(1, 1))
    x = id_block(x,3,[64,64,256],stage=2,block='b')
    x = id_block(x, 3, [64,64,256], stage=2, block='c')

    x = conv_block(x,3,[128,128,512],stage=3,block ='a')
    x = id_block(x,3,[128,128,512],stage=3,block='b')
    x = id_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = id_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x,3,[256,256,1024],stage=4,block ='a')
    x = id_block(x,3,[256,256,1024],stage=4,block='b')
    x = id_block(x, 3, [256,256,1024], stage=4, block='c')
    x = id_block(x, 3, [256,256,1024], stage=4, block='d')
    x = id_block(x, 3, [256,256,1024], stage=4, block='e')
    x = id_block(x, 3, [256,256,1024], stage=4, block='f')

    x = conv_block(x,3,[512,512,2048],stage=5,block ='a')
    x = id_block(x,3,[512,512,2048],stage=5,block='b')
    x = id_block(x, 3, [512,512,2048], stage=5, block='c')
    print(x.shape)
    x = layers.AveragePooling2D((7,7),name='avg_pool')(x)
    print(x.shape)
    x = layers.Flatten()(x)
    x = layers.Dense(num_classes, activation='softmax', name='fc1000')(x)
    modle =Model(inputs=image, outputs=x,name='resnet50')
    modle.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    return modle

if __name__ == '__main__':
    model = ResNet50()
    model.summary()
    img_path = 'elephant.jpg'
    # img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
