# give an function to add pepper noise to an image
import cv2
import numpy as np
from skimage import util
def pepper_noise(image, amount):
    h, w, c = image.shape
    num_pepper = int(amount * h * w)
    for _ in range(num_pepper):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        image[y, x, :] = 0
    return image

# use the pepper noise API
img = cv2.imread('../imgs/Lenna.png')
pepper_img = util.random_noise(img, mode='pepper', amount=0.1)
cv2.imshow('Pepper Image', pepper_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

if __name__ == '__main__':
    img = cv2.imread('../imgs/Lenna.png')
    noise_img = pepper_noise(img, 0.1)
    cv2.imshow('Pepper Noise', noise_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
