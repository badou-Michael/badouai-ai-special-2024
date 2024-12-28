from lesson11_2_AlexNet_CNNbuild import AlexNet
import numpy as np
import cv2


if __name__ == '__main__':
    model = AlexNet().model
    model.load_weights(r'alexnet\AlexNet-Keras-master\logsepoch：15-loss：0.223-val_loss：0.291.h5')
    img = cv2.imread(r'alexnet\AlexNet-Keras-master\test2.jpg', cv2.COLOR_BGR2RGB)

    # 将宽高的值改为227，再做归一化
    img_norm = cv2.resize(img, (227, 227)) / 255.0
    # 在图片加上batch_size轴
    img_norm = np.expand_dims(img_norm, axis=0)

    # 开始用模型进行推理，获取推理结果
    prediction = model.predict(img_norm)
    print('prediction', prediction)
    # 获取正确率最高的值的索引
    pred_index = np.argmax(prediction)
    print('pred_index', pred_index)

    # 读取正确答案
    with open(r'alexnet\AlexNet-Keras-master\data\model\index_word.txt', 'r', encoding='utf-8') as f:
        ans = [i.split(';')[1][:1] for i in f.readlines()]
    # 打印正确答案
    print('answer：', ans[pred_index])










