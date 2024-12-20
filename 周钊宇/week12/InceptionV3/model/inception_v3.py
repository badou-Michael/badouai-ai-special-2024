import keras
from keras.layers import Conv2D, BatchNormalization, Activation,concatenate,Dense,Input,MaxPooling2D,AveragePooling2D,GlobalAveragePooling2D
from keras.models import Model


def ConvBnActiva(input_tensor, kernel_size, filter, padding='same',strides=(1,1)):

    x = Conv2D(filters=filter, kernel_size=kernel_size, strides=strides, padding=padding,use_bias=False)(input_tensor)
    x = BatchNormalization(scale=False)(x)
    x = Activation('relu')(x)

    return x

def InceptionV3(input_shape=[299,299,3], num_classes = 1000):

    input_tensor = Input(input_shape)
    x = ConvBnActiva(input_tensor, kernel_size=(3,3), filter=32, strides=(2,2), padding='valid')
    x = ConvBnActiva(x, kernel_size=(3,3), filter=32, padding='valid')
    x = ConvBnActiva(x, kernel_size=(3,3), filter=64)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    x = ConvBnActiva(x, kernel_size=(1,1), filter=80)
    x = ConvBnActiva(x, kernel_size=(3,3), filter=192, padding='valid')
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    #Block1 Part1

    branch1 = ConvBnActiva(x, kernel_size=(1,1), filter=64)

    branch2 = ConvBnActiva(x, kernel_size=(1,1), filter=48)
    branch2 = ConvBnActiva(branch2, kernel_size=(5,5), filter=64)

    branch3 = ConvBnActiva(x, kernel_size=(1,1), filter=64)
    branch3 = ConvBnActiva(branch3, kernel_size=(3,3), filter=96)
    branch3 = ConvBnActiva(branch3, kernel_size=(3,3), filter=96)

    branch4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
    branch4 = ConvBnActiva(branch4, kernel_size=(1,1), filter=32)

    block1_part1 = concatenate([branch1, branch2, branch3, branch4], axis=3)


    #Block1 part2
    branch1 = ConvBnActiva(block1_part1, kernel_size=(1,1), filter=64)

    branch2 = ConvBnActiva(block1_part1, kernel_size=(1,1), filter=48)
    branch2 = ConvBnActiva(branch2, kernel_size=(5,5), filter=64)

    branch3 = ConvBnActiva(block1_part1, kernel_size=(1,1), filter=64)
    branch3 = ConvBnActiva(branch3, kernel_size=(3,3), filter=96)
    branch3 = ConvBnActiva(branch3, kernel_size=(3,3), filter=96)

    branch4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(block1_part1)
    branch4 = ConvBnActiva(branch4, kernel_size=(1,1), filter=64)

    block1_part2 = concatenate([branch1, branch2, branch3, branch4], axis=3)


    #Block1 part3
    branch1 = ConvBnActiva(block1_part2, (1,1), filter=64)

    branch2 = ConvBnActiva(block1_part2, kernel_size=(1,1), filter=48)
    branch2 = ConvBnActiva(branch2, kernel_size=(5,5), filter=64)

    branch3 = ConvBnActiva(block1_part2, kernel_size=(1,1), filter=64)
    branch3 = ConvBnActiva(branch3, kernel_size=(3,3), filter=96)
    branch3 = ConvBnActiva(branch3, kernel_size=(3,3), filter=96)

    branch4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(block1_part2)
    branch4 = ConvBnActiva(branch4, kernel_size=(1,1), filter=64)

    block1_part3 = concatenate([branch1, branch2, branch3, branch4], axis=3)

    
    #Block2 part1
    branch1 = ConvBnActiva(block1_part3, kernel_size=(3,3), filter=384, padding='valid', strides=(2,2))

    branch2 = ConvBnActiva(block1_part3, kernel_size=(1,1), filter=64)
    branch2 = ConvBnActiva(branch2, kernel_size=(3,3), filter=96)
    branch2 = ConvBnActiva(branch2, kernel_size=(3,3), filter=96, padding='valid', strides=(2,2))

    branch3 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(block1_part3)

    block2_part1 = concatenate([branch1, branch2, branch3], axis=3)



    #Block2 part2
    branch1 = ConvBnActiva(block2_part1, kernel_size=(1,1), filter=192)

    branch2 = ConvBnActiva(block2_part1, kernel_size=(1,1), filter=128)
    branch2 = ConvBnActiva(branch2, kernel_size=(1,7), filter=128)
    branch2 = ConvBnActiva(branch2, kernel_size=(7,1), filter=192)

    branch3 = ConvBnActiva(block2_part1, kernel_size=(1,1), filter=128)
    branch3 = ConvBnActiva(branch3, kernel_size=(7,1), filter=128)
    branch3 = ConvBnActiva(branch3, kernel_size=(1,7), filter=128)
    branch3 = ConvBnActiva(branch3, kernel_size=(7,1), filter=128)
    branch3 = ConvBnActiva(branch3, kernel_size=(1,7), filter=192)

    branch4 = AveragePooling2D(pool_size=(3,3), strides=(1,1),padding='same')(block2_part1)
    branch4 = ConvBnActiva(branch4, kernel_size=(1,1), filter=192)

    block2_part2 = concatenate([branch1, branch2, branch3, branch4], axis=3)


    #Block2 part3&4
    branch1 = ConvBnActiva(block2_part2, kernel_size=(1,1), filter=192)

    branch2 = ConvBnActiva(block2_part2, kernel_size=(1,1), filter=160)
    branch2 = ConvBnActiva(branch2, kernel_size=(1,7), filter=160)
    branch2 = ConvBnActiva(branch2, kernel_size=(7,1), filter=192)

    branch3 = ConvBnActiva(block2_part2, kernel_size=(1,1), filter=160)
    branch3 = ConvBnActiva(branch3, kernel_size=(7,1), filter=160)
    branch3 = ConvBnActiva(branch3, kernel_size=(1,7), filter=160)
    branch3 = ConvBnActiva(branch3, kernel_size=(7,1), filter=160)
    branch3 = ConvBnActiva(branch3, kernel_size=(1,7), filter=192)

    branch4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(block2_part2)
    branch4 = ConvBnActiva(branch4, kernel_size=(1,1), filter=192)

    block2_part3 = concatenate([branch1, branch2, branch3, branch4], axis=3)

    branch1 = ConvBnActiva(block2_part3, kernel_size=(1,1), filter=192)

    branch2 = ConvBnActiva(block2_part3, kernel_size=(1,1), filter=160)
    branch2 = ConvBnActiva(branch2, kernel_size=(1,7), filter=160)
    branch2 = ConvBnActiva(branch2, kernel_size=(7,1), filter=192)

    branch3 = ConvBnActiva(block2_part3, kernel_size=(1,1), filter=160)
    branch3 = ConvBnActiva(branch3, kernel_size=(7,1), filter=160)
    branch3 = ConvBnActiva(branch3, kernel_size=(1,7), filter=160)
    branch3 = ConvBnActiva(branch3, kernel_size=(7,1), filter=160)
    branch3 = ConvBnActiva(branch3, kernel_size=(1,7), filter=192)

    branch4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(block2_part3)
    branch4 = ConvBnActiva(branch4, kernel_size=(1,1), filter=192)

    block2_part4 = concatenate([branch1, branch2, branch3, branch4], axis=3)


    #Block2 Part5
    branch1 = ConvBnActiva(block2_part4, kernel_size=(1,1), filter=192)

    branch2 = ConvBnActiva(block2_part4, kernel_size=(1,1), filter=192)
    branch2 = ConvBnActiva(branch2, kernel_size=(1,7), filter=192)
    branch2 = ConvBnActiva(branch2, kernel_size=(7,1), filter=192)

    branch3 = ConvBnActiva(block2_part4, kernel_size=(1,1), filter=192)
    branch3 = ConvBnActiva(branch3, kernel_size=(7,1),filter=192)
    branch3 = ConvBnActiva(branch3, kernel_size=(1,7), filter=192)
    branch3 = ConvBnActiva(branch3, kernel_size=(7,1),filter=192)
    branch3 = ConvBnActiva(branch3, kernel_size=(1,7), filter=192)

    branch4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(block2_part4)
    branch4 = ConvBnActiva(branch4, kernel_size=(1,1), filter=192)

    block2_part5 = concatenate([branch1, branch2, branch3, branch4], axis=3)


    #Block3 Part1
    branch1 = ConvBnActiva(block2_part5, kernel_size=(1,1), filter=192)
    branch1 = ConvBnActiva(branch1, kernel_size=(3,3), filter=320, padding='valid', strides=(2,2))

    branch2 = ConvBnActiva(block2_part5, kernel_size=(1,1), filter=192)
    branch2 = ConvBnActiva(branch2, kernel_size=(1,7), filter=192)
    branch2 = ConvBnActiva(branch2, kernel_size=(7,1), filter=192)
    branch2 = ConvBnActiva(branch2, kernel_size=(3,3), filter=192, strides=(2,2), padding='valid')

    branch3 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(block2_part5)

    block3_part1 = concatenate([branch1, branch2, branch3], axis=3)


    #Block3 Part2
    branch1 = ConvBnActiva(block3_part1, kernel_size=(1,1), filter=320)

    branch2 = ConvBnActiva(block3_part1, kernel_size=(1,1), filter=384)
    branch2_1 = ConvBnActiva(branch2, kernel_size=(1,3), filter=384)
    branch2_2 = ConvBnActiva(branch2, kernel_size=(3,1), filter=384)
    branch2 = concatenate([branch2_1, branch2_2], axis=3)

    branch3 = ConvBnActiva(block3_part1, kernel_size=(1,1), filter=448)
    branch3 = ConvBnActiva(branch3, kernel_size=(3,3), filter=384)
    branch3_1 = ConvBnActiva(branch3, kernel_size=(1,3), filter=384)
    branch3_2 = ConvBnActiva(branch3, kernel_size=(3,1), filter=384)
    branch3 = concatenate([branch3_1, branch3_2], axis=3)

    branch4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(block3_part1)
    branch4 = ConvBnActiva(branch4, kernel_size=(1,1), filter=192)

    block3_part2 = concatenate([branch1, branch2, branch3, branch4], axis=3)


    #Block3 Part3
    branch1 = ConvBnActiva(block3_part2, kernel_size=(1,1), filter=320)

    branch2 = ConvBnActiva(block3_part2, kernel_size=(1,1), filter=384)
    branch2_1 = ConvBnActiva(branch2, kernel_size=(1,3), filter=384)
    branch2_2 = ConvBnActiva(branch2, kernel_size=(3,1), filter=384)
    branch2 = concatenate([branch2_1, branch2_2], axis=3)

    branch3 = ConvBnActiva(block3_part2, kernel_size=(1,1), filter=448)
    branch3 = ConvBnActiva(branch3, kernel_size=(3,3), filter=384)
    branch3_1 = ConvBnActiva(branch3, kernel_size=(1,3), filter=384)
    branch3_2 = ConvBnActiva(branch3, kernel_size=(3,1), filter=384)
    branch3 = concatenate([branch3_1, branch3_2], axis=3)

    branch4 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(block3_part2)
    branch4 = ConvBnActiva(branch4, kernel_size=(1,1), filter=192)

    block3_part3 = concatenate([branch1, branch2, branch3, branch4], axis=3)


    #Dense
    x = GlobalAveragePooling2D()(block3_part3)
    x = Dense(num_classes, activation='softmax')(x)


    model = Model(inputs=input_tensor, outputs=x)
    # model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    return model
    

    
    


# model = InceptionV3()
# model.summary()



