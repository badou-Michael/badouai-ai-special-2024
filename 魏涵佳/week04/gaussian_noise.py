# give a gaussian noise function
import cv2
import numpy as np
from skimage import util

def gaussian_noise(image, mean, std):
    h, w, c = image.shape
    noise = np.random.normal(mean, std, (h, w, c))
    noise_img = image + noise
    noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
    return noise_img


# use the gaussian noise API
img = cv2.imread('../imgs/Lenna.png')
gaussian_img = cv2.GaussianBlur(img, (5, 5), 1)
cv2.imshow('Gaussian Image', gaussian_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# use the skimage API
img = cv2.imread('../imgs/Lenna.png')
noise_img = util.random_noise(img, mode='gaussian', mean=0, var=0.01)
cv2.imshow('Gaussian Noise', noise_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

if __name__ == '__main__':
    img = cv2.imread('../imgs/Lenna.png')
    noise_img = gaussian_noise(img, 0, 50)
    cv2.imshow('Gaussian Noise', noise_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
