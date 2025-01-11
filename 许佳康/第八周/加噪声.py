import cv2
import numpy as np

#泊松噪声（Poisson Noise）
#椒盐噪声（Salt and Pepper Noise）
#高斯噪声（Gaussian Noise）
#斑点噪声（Speckle Noise）

def add_noise(image, noise_type="gaussian", **kwargs):
    if noise_type == "gaussian":
        return add_gaussian_noise(image, **kwargs)
    elif noise_type == "salt_and_pepper":
        return add_salt_and_pepper_noise(image, **kwargs)
    elif noise_type == "poisson":
        return add_poisson_noise(image)
    elif noise_type == "speckle":
        return add_speckle_noise(image)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

def add_gaussian_noise(image, mean=0, var=0.01):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape).astype('float32')
    noisy = cv2.add(image.astype('float32'), gauss)
    return np.clip(noisy, 0, 255).astype('uint8')

def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy = image.copy()
    total_pixels = image.size // image.shape[2]
    num_salt = np.ceil(salt_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 255
    num_pepper = np.ceil(pepper_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 0
    return noisy

def add_poisson_noise(image):
    noisy = np.random.poisson(image).astype('float32')
    return np.clip(noisy, 0, 255).astype('uint8')

def add_speckle_noise(image):
    gauss = np.random.randn(*image.shape).astype('float32')
    noisy = image.astype('float32') + image.astype('float32') * gauss
    return np.clip(noisy, 0, 255).astype('uint8')

# 示例
image = cv2.imread('../lenna.png')
noisy_gaussian = add_noise(image, noise_type="gaussian", mean=0, var=0.01)
noisy_salt_and_pepper = add_noise(image, noise_type="salt_and_pepper", salt_prob=0.01, pepper_prob=0.01)
noisy_poisson = add_noise(image, noise_type="poisson")
noisy_speckle = add_noise(image, noise_type="speckle")

# 保存结果
cv2.imwrite('image_gaussian_noise.jpg', noisy_gaussian)
cv2.imwrite('image_salt_and_pepper_noise.jpg', noisy_salt_and_pepper)
cv2.imwrite('image_poisson_noise.jpg', noisy_poisson)
cv2.imwrite('image_speckle_noise.jpg', noisy_speckle)
