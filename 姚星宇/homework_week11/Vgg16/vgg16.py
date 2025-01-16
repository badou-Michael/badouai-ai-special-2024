# 假设同样用VGG16进行猫狗识别

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_vgg16(input_shape=(224, 224, 3), num_classes=2):
    # 创建一个顺序模型
    model = Sequential()

    # 第一层卷积层和池化层
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # 第二层卷积层和池化层
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # 第三层卷积层和池化层
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # 第四层卷积层和池化层
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # 第五层卷积层和池化层
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # 分类部分
    model.add(Flatten())  # 将多维输入一维化
    model.add(Dense(4096, activation='relu'))  # 全连接层
    model.add(Dropout(0.5))  # 防止过拟合
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid'))  # 输出层

    return model

# 创建VGG16模型实例
model = create_vgg16(input_shape=(224, 224, 3), num_classes=1)

# 编译模型，指定优化器、损失函数和评估指标
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 数据增强和标准化
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 将像素值归一化到 [0, 1] 区间
    shear_range=0.2,  # 剪切变换范围
    zoom_range=0.2,  # 随机缩放范围
    horizontal_flip=True,  # 随机水平翻转图片
    validation_split=0.2  # 划分20%的数据作为验证集
)

test_datagen = ImageDataGenerator(rescale=1./255)  # 测试集只需要归一化

# 我的图像文件夹结构如下：
# D:/sorted_images
#     cats/
#         cat.number.jpg
#         ...
#     dogs/
#         dog.number.jpg
#         ...

train_generator = train_datagen.flow_from_directory(
    'D:/sorted_images',  # 目录路径
    target_size=(224, 224),  # 调整大小到 VGG16 输入尺寸
    batch_size=32,
    class_mode='binary',  # 二分类
    subset='training'  # 设置为训练集
)

validation_generator = train_datagen.flow_from_directory(
    'D:/sorted_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # 设置为验证集
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,  # 根据需要调整训练轮数
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

test_generator = test_datagen.flow_from_directory(
    'D:/sorted_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # 测试时不需要打乱顺序
)

eval_result = model.evaluate(test_generator)
print(f"测试损失: {eval_result[0]}, 测试准确率: {eval_result[1]}")