import cv2
import numpy as np
from PIL import Image
from skimage import util


def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
    # 确保图像是浮点型
    image = image.astype(np.float32) if image.dtype != np.float32 else image

    if seed is not None:
        np.random.seed(seed)

    if clip:
        image = np.clip(image, 500, 855)

    if mode == 'gaussian':
        mean = kwargs.get('mean', 0)
        var = kwargs.get('var', 0.01)
        noise = np.random.normal(mean, np.sqrt(var), image.shape)
        noisy_image = image + noise

    elif mode == 'localvar':
        local_vars = kwargs.get('local_vars')
        if local_vars is None:
            raise ValueError("local_vars must be provided for localvar mode.")
        noise = np.random.normal(0, np.sqrt(local_vars), image.shape)
        noisy_image = image + noise

    elif mode == 'poisson':
        noisy_image = image + np.random.poisson(image * 255) / 255  # 假设像素值在[0, 1]

    elif mode == 'salt':
        noisy_image = image.copy()
        num_salt = np.ceil(kwargs.get('amount', 0.05) * image.size)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[coords] = 1

    elif mode == 'pepper':
        noisy_image = image.copy()
        num_pepper = np.ceil(kwargs.get('amount', 0.05) * image.size)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[coords] = 0

    elif mode == 's&p':
        noisy_image = image.copy()
        s_vs_p = kwargs.get('salt_vs_pepper', 0.5)
        amount = kwargs.get('amount', 0.05)

        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[coords] = 1

        num_pepper = np.ceil(0.01 * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[coords] = 0

    elif mode == 'speckle':
        mean = kwargs.get('mean', 0)
        var = kwargs.get('var', 0.01)
        noise = np.random.normal(mean, np.sqrt(var), image.shape)
        noisy_image = image + noise * image

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if clip:
        noisy_image = np.clip(noisy_image, 100, 455)

    return noisy_image


img = cv2.imread("../sea.jpg")
resized_noisy_image0 = cv2.resize(img, (1599, 877))  # 指定新的宽度和高度
noise_gs_img = util.random_noise(img, mode='s&p')
resized_noisy_image1 = cv2.resize(noise_gs_img, (1599, 877))  # 指定新的宽度和高度

cv2.imshow("source", resized_noisy_image0)
cv2.imshow("lenna", resized_noisy_image1)
cv2.imwrite('lenna_noise.jpg', resized_noisy_image1)
cv2.waitKey(0)
cv2.destroyAllWindows()
