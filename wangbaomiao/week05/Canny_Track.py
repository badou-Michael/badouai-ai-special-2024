# -*- coding: utf-8 -*-
# time: 2024/10/26 16:12
# file: Canny_Track.py
# author: flame
import cv2

"""
详细解释

程序概述：
该程序使用 OpenCV 库实现 Canny 边缘检测，并通过创建滑动条来动态调整 Canny 算法的高低阈值，从而观察不同阈值下边缘检测的效果。

函数 CannyThreshold：
    参数：
        lowThreshold: Canny 算法的低阈值，用于检测弱边缘。
        highThreshold: Canny 算法的高阈值，用于检测强边缘。
    
    功能：
        使用 Canny 算法进行边缘检测。
        将检测到的边缘与原图进行位运算，以显示边缘检测的结果。
        显示 Canny 边缘检测的结果。

变量初始化：
    lowThreshold = 0: 初始化低阈值为 0。
    highThreshold = 100: 初始化高阈值为 100。
    maxThreshold = 100: 设置高阈值的最大值为 100，用于界面中的调节杠。
    ratio = 3: 阈值比率，用于计算 Canny 算法的高阈值。
    kernel_size = 3: Canny 算法使用的 Sobel 算子的大小。

图像处理：
    img = cv2.imread("lenna.png"): 读取图像文件。
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY): 将图像转换为灰度图，以便进行边缘检测。

窗口和滑动条：
    cv2.namedWindow("Canny result"): 创建一个窗口用于展示边缘检测的结果。
    cv2.createTrackbar("Min threshold", "Canny result", lowThreshold, maxThreshold, 
                       lambda x: CannyThreshold(x, highThreshold)): 
        设置低阈值调节杠，允许用户调整 Canny 算法的低阈值。
    
    cv2.createTrackbar("Max threshold", "Canny result", highThreshold, maxThreshold, 
                       lambda x: CannyThreshold(lowThreshold, x)): 
        设置高阈值调节杠，允许用户调整 Canny 算法的高阈值。

初始调用和用户交互：
    CannyThreshold(lowThreshold, highThreshold): 初始调用 CannyThreshold 函数，设置低阈值为 0，高阈值为 100。
    
    if cv2.waitKey(0) == 27: 
        cv2.destroyAllWindows(): 等待用户操作，如果用户按下 ESC 键，则关闭所有窗口。

通过这些详细的注释和解释，可以更好地理解程序的逻辑和每一步的操作。

该程序使用OpenCV库实现Canny边缘检测，并通过创建滑动条来动态调整Canny算法的高低阈值，从而观察不同阈值下边缘检测的效果。
1. 读取图像并将其转换为灰度图。
2. 创建一个窗口用于显示边缘检测的结果。
3. 创建两个滑动条分别用于调整Canny算法的低阈值和高阈值。
4. 定义`CannyThreshold`函数，该函数根据当前的高低阈值执行Canny边缘检测，并将结果与原图进行位运算后显示。
5. 初始调用`CannyThreshold`函数，并等待用户操作，如果用户按下ESC键，则关闭所有窗口。
"""

def CannyThreshold(lowThreshold, highThreshold):
    """
    使用Canny算法进行边缘检测，并显示结果。

    参数:
    - lowThreshold: Canny算法的低阈值
    - highThreshold: Canny算法的高阈值
    """
    # 使用Canny算法进行边缘检测
    detected_edges = cv2.Canny(img_gray, lowThreshold, highThreshold, kernel_size)
    # 将检测到的边缘与原图进行位运算，以显示边缘检测的结果
    dst = cv2.bitwise_and(img, img, mask=detected_edges)
    # 显示Canny边缘检测的结果
    cv2.imshow("Canny result", dst)

# 初始化低阈值为0
lowThreshold = 0
# 初始化高阈值为100
highThreshold = 100
# 设置高阈值的最大值为100，用于界面中的调节杠
maxThreshold = 100
# 阈值比率，用于计算Canny算法的高阈值
ratio = 3
# Canny算法使用的Sobel算子的大小
kernel_size = 3

# 读取图像文件
img = cv2.imread("lenna.png")
# 将图像转换为灰度图，以便进行边缘检测
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建一个窗口用于展示边缘检测的结果
cv2.namedWindow("Canny result")

# 设置低阈值调节杠，允许用户调整Canny算法的低阈值
cv2.createTrackbar("Min threshold", "Canny result", lowThreshold, maxThreshold, lambda x: CannyThreshold(x, highThreshold))
# 设置高阈值调节杠，允许用户调整Canny算法的高阈值
cv2.createTrackbar("Max threshold", "Canny result", highThreshold, maxThreshold, lambda x: CannyThreshold(lowThreshold, x))

# 初始调用CannyThreshold函数，设置低阈值为0，高阈值为100
CannyThreshold(lowThreshold, highThreshold)

# 等待用户操作，如果按下ESC键，则关闭所有窗口
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

