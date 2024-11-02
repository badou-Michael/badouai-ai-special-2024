import cv2 
import numpy as np

def nearest_interp(img, target_height, target_width):
    height, width, channels = img.shape
    
    # 创建一个target_height*target_width的空白图像
    newImage = np.zeros((target_height, target_width, channels), np.uint8)
    
    # 计算比例
    h_ratio = target_height / height
    w_ratio = target_width / width
    
    # 遍历新图像的每个像素
    for i in range(target_height):
        for j in range(target_width):
            # 计算原图像的坐标
            x = int(i / h_ratio + 0.5)  #int(),转为整型，使用向下取整。
            y = int(j / w_ratio + 0.5)
            # 赋值
            newImage[i, j] = img[x, y]
    return newImage


# 读取原始图像
img=cv2.imread("lenna.png")

# 调用函数
dst=nearest_interp(img, 800, 800)
# 打印结果
print(dst)
print(dst.shape)

# 显示图像
cv2.imshow("nearest interp",dst)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
# 读取原始图像
img=cv2.imread("lenna.png")
# 使用OpenCV的resize接口实现相同的效果
resized_img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_NEAREST) # cv2.INTER_NEAREST: nearest interpolation

# 打印结果
print(resized_img)
print(resized_img.shape)

# 显示图像
cv2.imshow("nearest interp with OpenCV", resized_img)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
