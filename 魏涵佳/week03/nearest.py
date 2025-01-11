import cv2
import numpy as np

def nearest(img, scale_w, scale_h):
    h, w, c = img.shape

    # create a new image
    dist_h, dist_w = int(h * scale_h), int(w * scale_w)
    empty_img = np.zeros((dist_h, dist_w, c), dtype=np.uint8)

    for i in range(dist_h):
        for j in range(dist_w):
            # find the nearest pixel in the original image
            nearest_h, nearest_w = int(i / scale_h), int(j / scale_w)
            empty_img[i, j] = img[nearest_h, nearest_w]

    return empty_img


if __name__ == '__main__':
    # Load the image
    img = cv2.imread('../imgs/Lenna.png')
    resize_img = nearest(img, 1.5, 1.5)

    # display the image and the resized image
    cv2.imshow('Original Image', img)
    cv2.imshow('Nearest Image', resize_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
