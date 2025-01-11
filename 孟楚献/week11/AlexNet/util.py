import cv2
import numpy as np
import tensorflow as tf

def resize_images(images, shape):
    with tf.name_scope("image_resize"):
        ret_images = []
        for image in images:
            image = cv2.resize(image, shape)
            ret_images.append(image)
        return np.array(ret_images)

def print_answer(argmax):
    with open("./data/model/index_word.txt", "r", encoding='utf-8') as f:
        synset = [l.split(";")[1][:-1] for l in f.readlines()]

    # print(synset[argmax])
    return synset[argmax]