# 用 keras 实现 Alexnet


import os
import shutil
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 先将训练图片分类
source_dir = "D:/train"              # 所有猫狗的路径
target_base_dir = "D:/sorted_images"      # 目标文件夹路径

# # 目标文件夹结构
# target_dirs = { 'cat' : os.path.join(target_base_dir, 'cats'),
#                'dog' : os.path.join(target_base_dir, 'dogs') }
# for label, target_dir in target_dirs.items():
#     os.makedirs(target_dir, exist_ok=True)

# # 遍历源目录并移动图片
# for filename in os.listdir(source_dir):
#     if filename.startswith('cat'):
#         shutil.move(os.path.join(source_dir, filename), target_dirs['cat'])
#     elif filename.startswith('dog'):
#         shutil.move(os.path.join(source_dir, filename), target_dirs['dog'])
# print("Images have been moved!")

img_width, img_height = 227, 227
batch_size = 32

# 创建ImageDataGenerator实例用于数据增强和归一化
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 分割20%的数据作为验证集
)

# 加载训练数据
train_generator = train_datagen.flow_from_directory(
    target_base_dir,  # 主目录
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'  
)

# 加载验证数据
validation_generator = train_datagen.flow_from_directory(
    target_base_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  
)

def create_alexnet(input_shape=(227, 227, 3), num_classes=1):
    model = Sequential([
        Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
        Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
        Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')  # 使用sigmoid激活函数进行二分类
    ])
    return model

# 创建模型
model = create_alexnet()

model.compile(
    loss='binary_crossentropy',  # 二分类问题使用binary_crossentropy
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,  # 可根据需要调整训练轮数
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

model.save('alexnet_cats_vs_dogs.h5')
