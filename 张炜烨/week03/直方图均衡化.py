import cv2
import numpy as np

def histogram_equalization(image):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算灰度图像的直方图
    hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])
    
    # 计算累积分布函数（CDF）
    cdf = hist.cumsum()
    # 正规化累积分布函数，使其最大值为255
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    
    # 使用累积分布函数对图像进行变换
    cdf_m = np.ma.masked_equal(cdf_normalized, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # 应用变换
    image_equalized = cdf[gray_image]
    
    # 转换回BGR颜色空间
    return cv2.cvtColor(image_equalized, cv2.COLOR_GRAY2BGR)

# 读取图像
img = cv2.imread('lenna.png')
# 检查图像是否成功读取
if img is None:
    print("Error: Image not found.")
else:
    # 应用直方图均衡化
    equalized_img = histogram_equalization(img)
    # 显示原始图像和均衡化后的图像
    # cv2.imshow('Original Image', img)
    # cv2.imshow('Histogram Equalized Image', equalized_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 保存均衡化后的图像
    cv2.imwrite('equalized_lenna.png', equalized_img)