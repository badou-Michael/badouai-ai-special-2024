from mask_rcnn import MASK_RCNN
from PIL import Image

mask_rcnn = MASK_RCNN()
image_path = "img/street.jpg"

while True:
    img = input(image_path)
    try:
        image = Image.open(image_path)
    except:
        print('Open Error! Try again!')
        continue
    else:
        mask_rcnn.detect_image(image)

mask_rcnn.close_session()