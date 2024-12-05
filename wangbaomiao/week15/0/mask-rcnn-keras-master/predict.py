# -*- coding: utf-8 -*-
# time: 2024/11/29 10:27
# file: predict.py
# author: flame
from mask_mrcnn import MASK_RCCN
from PIL import Image

mask_rcnn = MASK_RCCN()

while True:
    # 直接读取图像文件
    image_path = 'img/street.jpg'
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f'Error: File {image_path} not found.')
        continue
    except Exception as e:
        print(f'Error: {e}')
        continue
    else:
        mask_rcnn.detect_image(image)

mask_rcnn.close_session()