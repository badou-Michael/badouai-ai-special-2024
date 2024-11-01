# 实现透视变换（Perspective Transformation）

import cv2
import numpy as np

def perspective_transform(img):

    # 读取图像
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用Canny边缘检测算法检测图像中的边缘
    edges = cv2.Canny(gray, 80, 150)

    # 寻找最大的轮廓
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    '''
    cv2.RETR_EXTERNAL：只检测外轮廓
    cv2.CHAIN_APPROX_SIMPLE：压缩轮廓，将冗余的点去掉，从而减少内存消耗，只保留轮廓的端点
    '''
    contour = max(contours, key=cv2.contourArea) # cv2.contourArea()函数计算轮廓的面积

    # 获取轮廓的四个顶点
    epsilon = 0.076 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 确保是四个点
    if len(approx) == 4:
        points_src = np.float32(approx.reshape(4, 2))
        width = int(max(np.linalg.norm(points_src[0] - points_src[1]), np.linalg.norm(points_src[2] - points_src[3])))
        height = int(max(np.linalg.norm(points_src[0] - points_src[3]), np.linalg.norm(points_src[1] - points_src[2])))
        points_dst = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

        # 透视变换
        M = cv2.getPerspectiveTransform(points_src, points_dst)
        img_warp = cv2.warpPerspective(img, M, (width, height))

        # 显示结果
        cv2.imshow('Original Image', img)
        cv2.imshow('Warped Image', img_warp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('Not enough points to apply perspective transform')


if __name__ == '__main__':
    img = 'A4.jpg'
    perspective_transform(img)
