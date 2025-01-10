import os
import week14_yolo_config
import argparse
import numpy as np
import tensorflow as tf
import week14_yolo_predict as yolo_predictor
from PIL import Image, ImageFont, ImageDraw
from week14_yolo_utils import letterbox_image, load_weights

os.environ["CUDA_VISIBLE_DEVICES"]=week14_yolo_config.gpu_index #指定使用gpu的index


def detect(image_path,model_path,yolo_weights=None):
    '''加载模型进行预测，'''

    #先对图片进行预处理
    image=Image.open(image_path)
    #调用letterbox_image函数对输入的图像进行缩放，按照长宽比缩放，不足的部分用灰色填充
    resize_image=letterbox_image(image,(416,416))
    image_data=np.array(resize_image,dtype=np.float32)#转换为numpy数组
    #归一化
    image_data=image_data/255
    #使用NumPy的expand_dims函数在image_data数组的第一个维度（axis=0）上添加一个新的维度。
    # 这样做通常是为了将单个图像的数组从形状(416, 416, 3)变为(1, 416, 416, 3)，使其成为一个批量（batch）的图像。
    # 这是因为大多数深度学习框架期望输入数据是一个四维数组，其中第一个维度是批量大小。
    image_data=np.expand_dims(image_data,axis=0)

    #图片输入
    #创建一个占位符input_image_shape，用于在后续的计算图中接收图像的原始尺寸
    #shape=(2,)指定了占位符的形状为一个包含两个元素的一维数组，通常用来存储图像的高度和宽度
    input_image_shape=tf.placeholder(dtype=tf.int32,shape=(2,))

    #创建另一个占位符input_image，用于接收输入的图像数据
    #shape=[None, 416, 416, 3]表示这个占位符可以接收任意数量的图像，每个图像的尺寸为416x416像素，且为三通道
    input_image=tf.placeholder(dtype=tf.float32,shape=[None,416,416,3])

    #进入yolo_predictor进行预测
    predictor=yolo_predictor(week14_yolo_config.obj_threshold,week14_yolo_config.nms_threshold,
                             week14_yolo_config.classes_path,week14_yolo_config.anchors_path)

    with tf.Session() as sess:
        #图片预测
        if yolo_weights is not None:
            #在TensorFlow中创建一个变量作用域（predict），用于管理变量的命名空间
            with tf.variable_scope('predict'):
                #调用predictor对象的predict方法来预测边界框、得分和类别
                boxes,scores,classes=predictor.predict(input_image,input_image_shape)
            #加载YOLO权重到模型中
            load_op=load_weights(tf.global_variables(scope='predict'),weights_file=yolo_weights)
            #执行权重加载操作
            sess.run(load_op)
            #运行预测，并将输入图像数据和图像尺寸传递给模型
            out_boxes,out_scores,out_classes=sess.run([boxes,scores,classes],
                                                      feed_dict={
                                                          input_image:image_data,
                                                          input_image_shape:[image.size[1],image.size[0]]
                                                      })
        else:
            #如果没有提供YOLO权重文件：直接使用predictor对象进行预测，并使用saver.restore来恢复模型。
            boxes,scores,classes=predictor.predict(input_image,input_image_shape)
            saver=tf.train.Saver()
            saver.restore(sess,model_path)
            out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes],
                                                          feed_dict={
                                                              input_image: image_data,
                                                              input_image_shape: [image.size[1], image.size[0]]
                                                          })

        #在图像上绘制检测到的目标的边界框和类别标签
        #打印检测到的边界框数量
        print('Found {} boxes for {}'.format(len(out_boxes),'img'))
        #设置字体
        font=ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2*image.size[1]+0.5).astype('int32'))
        #设置边界框的厚度
        thickness=(image.size[0]+image.size[1])//300

        #从后向前遍历检测到的类别索引，这样可以在绘制时避免覆盖
        for i,c in reversed(list(enumerate(out_classes))):
            #获得预测的类别、box和分数
            predicted_class=predictor.class_names[c]
            box=out_boxes[i]
            score=out_scores[i]
            #创建一个包含类别名称和置信度分数的标签
            label='{}{:.2f}'.format(predicted_class,score)

            #绘制边界框和标签
            draw=ImageDraw.Draw(image)
            #textsize用于获得写字的时候，按照这个字体，要多大的框
            label_size=draw.textsize(label,font)

            #获取边界框的坐标
            top,left,bottom,right=box
            top=max(0,np.floor(top+0.5).astype('int32'))
            left=max(0,np.floor(left+0.5).astype('int32'))
            bottom=min(image.size[1]-1,np.floor(bottom+0.5).astype('int32'))
            right=min(image.size[0]-1,np.floor(right+0.5).astype('int32'))
            print(label,(left,top),(right,bottom))
            print(label_size)

            #如果边界框的顶部有足够的空间，将文本放置在边界框上方；否则，放置在边界框内部
            if top-label_size[1]>=0:
                text_origin=np.array([left,top-label_size[1]])
            else:
                text_origin=np.array([left,top+1])

            #绘制边界框和文本
            for i in range(thickness):
                draw.rectangle([left+i,top+i,right-i,bottom-i],
                               outline=predictor.colors[c])
            draw.rectangle(
                [tuple(text_origin),tuple(text_origin+label_size)],
                fill=predictor.colors[c])
            draw.text(text_origin,label,fill=(0,0,0),font=font)
            del draw
        image.show()
        image.save('./img/result1,jpg')

if __name__=='__main__':
    if week14_yolo_config.pre_train_yolo3==True:
        detect(week14_yolo_config.image_file,week14_yolo_config.model_dir,week14_yolo_config.yolo3_weights_path)
    else:
        detect(week14_yolo_config.image_file,week14_yolo_config.model_dir)
