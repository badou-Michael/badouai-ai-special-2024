import numpy as np
import cv2
from skimage import util


def addgaussnoise(img, mean, sigma, gauss_p):
    """添加高斯噪声"""
    assert gauss_p <= 1 and gauss_p >= 0
    img_temp = img.copy().astype(np.int16)
    height, width, _ = img.shape

    random_indices_gauss = np.random.rand(height, width)
    gauss_noise = np.round(np.random.normal(mean, sigma, (height, width))).astype(
        np.int16
    )
    # 噪声掩码
    mask_gauss = random_indices_gauss < gauss_p
    # print(mask_gauss)
    img_temp += gauss_noise[:, :, None] * mask_gauss[:, :, None]
    # 限定像素范围
    img_temp = np.clip(img_temp, 0, 255).astype(np.uint8)

    return img_temp


def addpsnoise(img, p_pepper, p_salt):
    """添加椒盐噪声"""
    assert p_salt <= 1 and p_salt >= 0
    assert p_pepper <= 1 and p_pepper >= 0

    img_temp = img.copy()
    height, width, _ = img.shape

    random_indices_pepper = np.random.rand(height, width)
    random_indices_salt = np.random.rand(height, width)
    # 噪声掩码
    mask_papper = random_indices_pepper < p_pepper
    mask_salt = random_indices_salt < p_salt
    # print(mask_salt,mask_papper)
    img_temp[mask_salt] = 255
    img_temp[mask_papper] = 0

    return img_temp


img = cv2.imread("practice/cv/week04/lenna.png")

print(img.shape, img.dtype)

new_img_1 = addgaussnoise(img, mean=5, sigma=5, gauss_p=0.1)
new_img_2 = addpsnoise(img, 0.1, 0.1)

# 调用接口
new_img_3 = (util.random_noise(img, "poisson") * 255).astype(np.uint8)

cv2.imshow("img", np.hstack([img, new_img_1, new_img_2,new_img_3]))
# cv2.imshow("img", np.hstack(new_img_3))
cv2.waitKey(0)
