# 1. 高斯噪声

import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_noise(img, mean, sigma):
    '''
    此函数用将产生的高斯噪声加到图片上
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
        noise        : 对应的噪声
    '''
    # 将图片灰度标准化
    img = img / 255
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img.shape)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out*255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return gaussian_out, noise # 这里也会返回噪声，注意返回值

# 读取图片
src = cv2.imread('Lenna.jpg', 0)
# 创建绘图 figure
fig_out = plt.figure(figsize=(4, 2), dpi=370)  # figsize宽高比
fig_noise = plt.figure(figsize=(4, 2), dpi=370)

for i in range(0, 8):
    # 将图片和不同的噪声叠加
    gaussian_out, noise = gaussian_noise(src, 0, 0.03 * i)
    # 创建 AxesSubplot 对象
    ax_out = fig_out.add_subplot(2, 4, i + 1)
    ax_noise = fig_noise.add_subplot(2, 4, i + 1) # 也可以写成(i + 241
    # 将丑兮兮的坐标抽去掉
    ax_out.axis('off')
    ax_noise.axis('off')
    # 设置标题
    ax_out.set_title('$\sigma$ = ' + str(0.03 * i), loc='left', fontsize=3, fontstyle='italic')
    ax_noise.set_title('$\sigma$ = ' + str(0.03 * i), loc='left', fontsize=3, fontstyle='italic')
    # 图片展示
    ax_out.imshow(gaussian_out, cmap='gray')
    ax_noise.imshow((noise + 1) / 2, cmap='gray')

# 保存图片
fig_out.savefig('1_Peppers_noise.png')
fig_noise.savefig('1_Guassion_noise.png')
# 图片显示
plt.show()

# 2. 椒盐噪声

import cv2
import numpy as np
import matplotlib.pyplot as plt


def saltpepper_noise(image, proportion):
    '''
    此函数用于给图片添加椒盐噪声
    image       : 原始图片
    proportion  : 噪声比例
    '''
    image_copy = image.copy()
    # 求得其高宽
    img_Y, img_X = image.shape
    # 噪声点的 X 坐标，创建相同个数的img_Y 和 img_X
    X = np.random.randint(img_X, size=(int(proportion * img_X * img_Y),))
    # 噪声点的 Y 坐标
    Y = np.random.randint(img_Y, size=(int(proportion * img_X * img_Y),))
    # 噪声点的坐标赋值
    ''' 这里的 [0, 255]表示产生的随机数是 0 或 255'''
    image_copy[Y, X] = np.random.choice([0, 255], size=(int(proportion * img_X * img_Y),))

    # 噪声容器
    '''创建一个与原始图像大小相同的噪声图板，初始值为 127（灰色），用于将噪声点的像素值赋予，展示噪声的分布'''
    sp_noise_plate = np.ones_like(image_copy) * 127
    # 将噪声给噪声容器
    sp_noise_plate[Y, X] = image_copy[Y, X]
    return image_copy, sp_noise_plate  # 这里也会返回噪声，注意返回值


# 读取图片
src = cv2.imread('Lenna.jpg', 0)

# 创建绘图 figure
fig_out = plt.figure(figsize=(4, 2), dpi=370)  # figsize宽高比
fig_noise = plt.figure(figsize=(4, 2), dpi=370)

for i in range(0, 8):
    # 将图片和不同的噪声叠加
    sp_out, noise = saltpepper_noise(src, 0.03 * i)
    # 创建 AxesSubplot 对象
    ax_out = fig_out.add_subplot(i + 241)
    ax_noise = fig_noise.add_subplot(i + 241)
    # 将丑兮兮的坐标抽去掉
    ax_out.axis('off')
    ax_noise.axis('off')
    # 设置标题
    ax_out.set_title('proportion = ' + str(0.03 * i), loc='left', fontsize=3, fontstyle='italic')
    ax_noise.set_title('proportion = ' + str(0.03 * i), loc='left', fontsize=3, fontstyle='italic')
    # 图片展示
    ax_out.imshow(sp_out, cmap='gray')
    ax_noise.imshow(noise, cmap='gray')

# 保存图片
fig_out.savefig('1_Peppers_spnoise.png')
fig_noise.savefig('1_sp_noise.png')
# 图片显示
plt.show()
