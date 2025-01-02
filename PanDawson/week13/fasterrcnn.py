import cv2
import keras
import numpy as np
import colorsys
import pickle
import os
import nets.frcnn as frcnn
from nets.frcnn_training import get_new_img_size
from keras import backend as K
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image,ImageFont, ImageDraw
from utils.utils import BBoxUtility
from utils.anchors import get_anchors
from utils.config import Config
import copy
import math

from __future__ import division
from nets.frcnn import get_model
from nets.frcnn_training import cls_loss,smooth_l1,Generator,get_img_output_length,class_loss_cls,class_loss_regr

from utils.config import Config
from utils.utils import BBoxUtility
from utils.roi_helpers import calc_iou

from keras.utils import generic_utils
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import keras
import numpy as np
import time 
import tensorflow as tf
from utils.anchors import get_anchors


class FRCNN(object):
    _defaults = {
        "model_path": 'model_data/voc_weights.h5',
        "classes_path": 'model_data/voc_classes.txt',       
        "confidence": 0.7,   #置信度
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化faster RCNN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.sess = K.get_session() # Keras 会话
        self.config = Config() # RPN网络的参数
        self.generate()
        self.bbox_util = BBoxUtility()
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        # 计算总的种类
        self.num_classes = len(self.class_names)+1

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        self.model_rpn,self.model_classifier = frcnn.get_predict_model(self.config,self.num_classes)
        self.model_rpn.load_weights(self.model_path,by_name=True)
        self.model_classifier.load_weights(self.model_path,by_name=True,skip_mismatch=True)
                
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
    
    def get_img_output_length(self, width, height):
        def get_output_length(input_length):
            # input_length += 6
            filter_sizes = [7, 3, 1, 1]
            padding = [3,1,0,0]
            stride = 2
            for i in range(4):
                # input_length = (input_length - filter_size + stride) // stride
                input_length = (input_length+2*padding[i]-filter_sizes[i]) // stride + 1
            return input_length
        return get_output_length(width), get_output_length(height) 
    
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        old_width = image_shape[1]
        old_height = image_shape[0]
        old_image = copy.deepcopy(image)  #深拷贝
        width,height = get_new_img_size(old_width,old_height)


        image = image.resize([width,height])
        photo = np.array(image,dtype = np.float64)

        # 图片预处理，归一化
        photo = preprocess_input(np.expand_dims(photo,0))
        preds = self.model_rpn.predict(photo)
        # 将预测结果进行解码
        anchors = get_anchors(self.get_img_output_length(width,height),width,height)

        rpn_results = self.bbox_util.detection_out(preds,anchors,1,confidence_threshold=0)
        R = rpn_results[0][:, 2:]
        
        R[:,0] = np.array(np.round(R[:, 0]*width/self.config.rpn_stride),dtype=np.int32)
        R[:,1] = np.array(np.round(R[:, 1]*height/self.config.rpn_stride),dtype=np.int32)
        R[:,2] = np.array(np.round(R[:, 2]*width/self.config.rpn_stride),dtype=np.int32)
        R[:,3] = np.array(np.round(R[:, 3]*height/self.config.rpn_stride),dtype=np.int32)
        
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
        base_layer = preds[2]
        
        delete_line = []
        for i,r in enumerate(R):
            if r[2] < 1 or r[3] < 1:
                delete_line.append(i)
        R = np.delete(R,delete_line,axis=0)
        
        bboxes = []
        probs = []
        labels = []
        for jk in range(R.shape[0]//self.config.num_rois + 1):
            ROIs = np.expand_dims(R[self.config.num_rois*jk:self.config.num_rois*(jk+1), :], axis=0)
            
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//self.config.num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],self.config.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded
            
            [P_cls, P_regr] = self.model_classifier.predict([base_layer,ROIs])

            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :]) < self.confidence or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                label = np.argmax(P_cls[0, ii, :])

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])

                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= self.config.classifier_regr_std[0]
                ty /= self.config.classifier_regr_std[1]
                tw /= self.config.classifier_regr_std[2]
                th /= self.config.classifier_regr_std[3]

                cx = x + w/2.
                cy = y + h/2.
                cx1 = tx * w + cx
                cy1 = ty * h + cy
                w1 = math.exp(tw) * w
                h1 = math.exp(th) * h

                x1 = cx1 - w1/2.
                y1 = cy1 - h1/2.

                x2 = cx1 + w1/2
                y2 = cy1 + h1/2

                x1 = int(round(x1))
                y1 = int(round(y1))
                x2 = int(round(x2))
                y2 = int(round(y2))

                bboxes.append([x1,y1,x2,y2])
                probs.append(np.max(P_cls[0, ii, :]))
                labels.append(label)

        if len(bboxes)==0:
            return old_image
        
        # 筛选出其中得分高于confidence的框
        labels = np.array(labels)
        probs = np.array(probs)
        boxes = np.array(bboxes,dtype=np.float32)
        boxes[:,0] = boxes[:,0]*self.config.rpn_stride/width
        boxes[:,1] = boxes[:,1]*self.config.rpn_stride/height
        boxes[:,2] = boxes[:,2]*self.config.rpn_stride/width
        boxes[:,3] = boxes[:,3]*self.config.rpn_stride/height
        results = np.array(self.bbox_util.nms_for_out(np.array(labels),np.array(probs),np.array(boxes),self.num_classes-1,0.4))
        
        top_label_indices = results[:,0]
        top_conf = results[:,1]
        boxes = results[:,2:]
        boxes[:,0] = boxes[:,0]*old_width
        boxes[:,1] = boxes[:,1]*old_height
        boxes[:,2] = boxes[:,2]*old_width
        boxes[:,3] = boxes[:,3]*old_height

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        
        thickness = (np.shape(old_image)[0] + np.shape(old_image)[1]) // width
        image = old_image
        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]

            left, top, right, bottom = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def close_session(self):
        self.sess.close()




def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

if __name__ == "__main__":
    config = Config()
    NUM_CLASSES = 21
    EPOCH = 100
    EPOCH_LENGTH = 2000
    bbox_util = BBoxUtility(overlap_threshold=config.rpn_max_overlap,ignore_threshold=config.rpn_min_overlap)
    annotation_path = '2007_train.txt'

    model_rpn, model_classifier,model_all = get_model(config,NUM_CLASSES)
    base_net_weights = "model_data/voc_weights.h5"
    
    model_all.summary()
    model_rpn.load_weights(base_net_weights,by_name=True)
    model_classifier.load_weights(base_net_weights,by_name=True)

    with open(annotation_path) as f: 
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    gen = Generator(bbox_util, lines, NUM_CLASSES, solid=True)
    rpn_train = gen.generate()
    log_dir = "logs"
    # 训练参数设置
    logging = TensorBoard(log_dir=log_dir)
    callback = logging
    callback.set_model(model_all)
    
    model_rpn.compile(loss={
                'regression'    : smooth_l1(),
                'classification': cls_loss()
            },optimizer=keras.optimizers.Adam(lr=1e-5)
    )
    model_classifier.compile(loss=[
        class_loss_cls, 
        class_loss_regr(NUM_CLASSES-1)
        ], 
        metrics={'dense_class_{}'.format(NUM_CLASSES): 'accuracy'},optimizer=keras.optimizers.Adam(lr=1e-5)
    )
    model_all.compile(optimizer='sgd', loss='mae')

    # 初始化参数
    iter_num = 0
    train_step = 0
    losses = np.zeros((EPOCH_LENGTH, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = [] 
    start_time = time.time()
    # 最佳loss
    best_loss = np.Inf
    # 数字到类的映射
    print('Starting training')

    for i in range(EPOCH):

        if i == 20:
            model_rpn.compile(loss={
                        'regression'    : smooth_l1(),
                        'classification': cls_loss()
                    },optimizer=keras.optimizers.Adam(lr=1e-6)
            )
            model_classifier.compile(loss=[
                class_loss_cls, 
                class_loss_regr(NUM_CLASSES-1)
                ], 
                metrics={'dense_class_{}'.format(NUM_CLASSES): 'accuracy'},optimizer=keras.optimizers.Adam(lr=1e-6)
            )
            print("Learning rate decrease")
        
        progbar = generic_utils.Progbar(EPOCH_LENGTH) 
        print('Epoch {}/{}'.format(i + 1, EPOCH))
        while True:
            if len(rpn_accuracy_rpn_monitor) == EPOCH_LENGTH and config.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, EPOCH_LENGTH))
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
            
            X, Y, boxes = next(rpn_train)
            
            loss_rpn = model_rpn.train_on_batch(X,Y)
            write_log(callback, ['rpn_cls_loss', 'rpn_reg_loss'], loss_rpn, train_step)
            P_rpn = model_rpn.predict_on_batch(X)
            height,width,_ = np.shape(X[0])
            anchors = get_anchors(get_img_output_length(width,height),width,height)
            
            # 将预测结果进行解码
            results = bbox_util.detection_out(P_rpn,anchors,1, confidence_threshold=0)
            
            R = results[0][:, 2:]

            X2, Y1, Y2, IouS = calc_iou(R, config, boxes[0], width, height, NUM_CLASSES)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue
            
            #np.where 筛选出所有满足条件的数据的索引
            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if len(neg_samples)==0:
                continue

            if len(pos_samples) < config.num_rois//2:
                selected_pos_samples = pos_samples.tolist()
            else:
                selected_pos_samples = np.random.choice(pos_samples, config.num_rois//2, replace=False).tolist()
            try:
                selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples), replace=False).tolist()
            except:
                selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples), replace=True).tolist()
            
            sel_samples = selected_pos_samples + selected_neg_samples
            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            write_log(callback, ['detection_cls_loss', 'detection_reg_loss', 'detection_acc'], loss_class, train_step)


            losses[iter_num, 0]  = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]
            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]


            train_step += 1
            iter_num += 1
            progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                  ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

            
            if iter_num == EPOCH_LENGTH:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if config.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                
                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                write_log(callback,
                        ['Elapsed_time', 'mean_overlapping_bboxes', 'mean_rpn_cls_loss', 'mean_rpn_reg_loss',
                        'mean_detection_cls_loss', 'mean_detection_reg_loss', 'mean_detection_acc', 'total_loss'],
                        [time.time() - start_time, mean_overlapping_bboxes, loss_rpn_cls, loss_rpn_regr,
                        loss_class_cls, loss_class_regr, class_acc, curr_loss],i)
                    
                
                if config.verbose:
                    print('The best loss is {}. The current loss is {}. Saving weights'.format(best_loss,curr_loss))
                if curr_loss < best_loss:
                    best_loss = curr_loss
                model_all.save_weights(log_dir+"/epoch{:03d}-loss{:.3f}-rpn{:.3f}-roi{:.3f}".format(i,curr_loss,loss_rpn_cls+loss_rpn_regr,loss_class_cls+loss_class_regr)+".h5")
                
                break
              
frcnn = FRCNN()

while True:
    img = input('img/street.jpg')
    try:
        image = Image.open('img/street.jpg')
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = frcnn.detect_image(image)
        r_image.show()
frcnn.close_session()
    
