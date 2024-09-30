import cv2
import numpy as np

def bilinear_interpolation(image):
    height, width, channels = image.shape
    new_image = np.zeros((700, 700, channels), dtype=np.uint8)
    rh, rw = height / 700, width / 700
    for i in range(700):
        for j in range(700):
            x,y=(i+0.5)*rh-0.5,(j+0.5)*rw-0.5
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = min(x0 + 1, height - 1)
            y1 = min(y0 + 1, width - 1)
            temp0 = (x1 - x) * image[x0, y0] + (x - x0) * image[x1, y0]
            temp1 = (x1 - x) * image[x0, y1] + (x - x0) * image[x1, y1]
            new_image[i,j] = (y1 - y) * temp0 + (y - y0) * temp1
    return new_image

if __name__ == '__main__':
    im = cv2.imread('lenna.png')
    new_im = bilinear_interpolation(im)
    cv2.imshow('image', im)
    cv2.imshow('new image', new_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
