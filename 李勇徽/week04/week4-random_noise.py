import cv2
from skimage import util
import matplotlib.pyplot as plt


if __name__ == '__main__':
    img = cv2.imread('lenna.png', 0)
    plt.subplot(2,2,1)
    plt.title('Orignal Image')
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    # add Poisson noise
    img_poisson = util.random_noise(img, mode="poisson", clip=True)
    plt.subplot(2,2,2)
    plt.title('Image with Poisson Noise')
    plt.imshow(img_poisson, cmap = 'gray')
    plt.axis('off')

    # add Gaussian noise
    img_gaussian = util.random_noise(img, mode="gaussian", clip=True)
    plt.subplot(2,2,3)
    plt.title('Image with Gaussian Noise')
    plt.imshow(img_gaussian, cmap = 'gray')
    plt.axis('off')

    # add PepperandSalt noise
    img_s_p = util.random_noise(img, mode="s&p", clip=True)
    plt.subplot(2,2,4)
    plt.title('Image with s&p Noise')
    plt.imshow(img_s_p, cmap = 'gray')
    plt.axis('off')

    plt.show()

