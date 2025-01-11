from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')

# 读取图像
img = cv2.imread('./lenna.png')

# 检查图像是否包含Alpha通道
if img.shape[2] == 4:
    # 丢弃Alpha通道
    img_rgb = img[..., :3]
else:
    img_rgb = img
# 将BGR图像转换为RGB图像
img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
# 灰度化
img_gray = rgb2gray(img_rgb)  # 确保只使用RGB通道
print(img_gray)
plt.imshow(img_gray, cmap='gray')
plt.axis('off')  # 不显示坐标轴
plt.savefig('lenna_gray_skimage.png', bbox_inches='tight', pad_inches=0)  # 保存灰度图像

# 二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("____img_binary____")
print(img_binary)
print(img_binary.shape)
plt.imshow(img_binary, cmap='gray')
plt.axis('off')  # 不显示坐标轴
plt.savefig('lenna_binary.png', bbox_inches='tight', pad_inches=0)  # 保存二值图像

plt.close()
################################################其他方法#################################
#加权平均法：
#这种方法通过给RGB通道不同的权重来计算灰度值。人眼对绿色的敏感度最高，其次是红色，对蓝色的敏感度最低。
from PIL import Image

# 打开图像
img = Image.open("lenna.png")

# 转换为灰度图像（加权平均法）
grayscale_img = img.convert("L")

# 保存灰度图像
grayscale_img.save("grayscale_image.png")
#平均值法：
#这种方法简单地将RGB三个通道的值相加后除以3。
from PIL import Image
import numpy as np

# 打开图像
img = Image.open("lenna.png")
img_array = np.array(img)

# 计算灰度值（平均值法）
grayscale_array = np.mean(img_array, axis=2)

# 转换回图像
grayscale_img = Image.fromarray(grayscale_array.astype('uint8'))

# 保存灰度图像
grayscale_img.save("grayscale_image_2.png")
# Luminosity 方法：
# 这种方法类似于加权平均法，但是权重略有不同，更接近于人眼对亮度的感知。
from PIL import Image
import numpy as np

# 打开图像
img = Image.open("lenna.png")
img_array = np.array(img)

# 计算灰度值（Luminosity 方法）
grayscale_array = 0.21*img_array[:,:,0] + 0.72*img_array[:,:,1] + 0.07*img_array[:,:,2]

# 转换回图像
grayscale_img = Image.fromarray(grayscale_array.astype('uint8'))

# 保存灰度图像
grayscale_img.save("grayscale_image_3.png")
