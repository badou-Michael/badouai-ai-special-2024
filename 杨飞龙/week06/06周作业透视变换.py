import cv2
import numpy as np

# 实现透视变换

# 定义鼠标回调函数 查看图片像素坐标
def show_pixel_value(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # 当鼠标移动时
        # 获取当前像素的颜色值（BGR格式）
        b, g, r = image[y, x]
        # 在图像窗口中显示像素的坐标和颜色值
        text = f"X: {x}, Y: {y}, R: {r}, G: {g}, B: {b}"
        img_copy = image.copy()  # 复制图像，用于展示实时信息
        cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Image', img_copy)

# 读取图片
image = cv2.imread('photo1.jpg')  # 替换为你要查看的图片路径
cv2.imshow('Image', image)

# 设置鼠标事件回调函数
cv2.setMouseCallback('Image', show_pixel_value)


# 透视变化
img = cv2.imread('photo1.jpg')
result3 = img.copy()

# 输入原图坐标，和想要转换后的坐标
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[17, 601], [343, 731], [207, 151], [517, 285]])


# 生成透视变换矩阵warp
m = cv2.getPerspectiveTransform(src, dst)

# 用warp矩阵和原图片叠加效果
result = cv2.warpPerspective(result3, m, (540, 960))

cv2.imshow("src", img)
cv2.imshow("result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
