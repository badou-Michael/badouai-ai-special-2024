import cv2
import numpy as np

# 定义一个全局变量存储点坐标
points = []


# 回调函数，用于鼠标点击事件
def get_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("src", img)
        if len(points) == 4:
            cv2.destroyWindow("src")


# 读取图片
img = cv2.imread('photo1.jpg')
result3 = img.copy()

# 创建窗口并绑定鼠标事件
cv2.imshow("src", img)
cv2.setMouseCallback("src", get_points)
cv2.waitKey(0)

if len(points) == 4:
    src = np.float32(points)
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

    print("Selected points:", src)

    # 生成透视变换矩阵；进行透视变换
    m = cv2.getPerspectiveTransform(src, dst)
    print("warpMatrix:")
    print(m)
    result = cv2.warpPerspective(result3, m, (337, 488))

    cv2.imshow("result", result)
    cv2.waitKey(0)
else:
    print("Error: You need to select exactly 4 points.")
