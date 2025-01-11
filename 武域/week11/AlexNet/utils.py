import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf

def load_image(path):
    img = mpimg.imread(path)
    short_edge = min(img.shape[:2])
    y = int((img.shape[0] - short_edge) / 2)
    x = int((img.shape[1] - short_edge) / 2)
    cropped = img[y:y+short_edge, x:x+short_edge]
    return cropped


def resized_image(image, size):
    with tf.name_scope('resized_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size, interpolation=cv2.INTER_AREA)
            images.append(i)
        images = np.array(images)
        return image


def print_answer(argmax):
    if argmax == 0:
        return "猫"  # cat in Chinese
    elif argmax == 1:
        return "狗"  # dog in Chinese
    else:
        return "Unknown"


