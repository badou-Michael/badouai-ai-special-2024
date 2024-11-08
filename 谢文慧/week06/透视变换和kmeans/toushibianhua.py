import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('photo1.jpg')
    '''
    注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
    '''
    # 原始图像坐标
    src = np.float32([[150, 151], [500, 250], [17, 500], [343, 731]])
    # 目标图像坐标
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    # 生成透视变换矩阵；进行透视变换
    m = cv2.getPerspectiveTransform(src,dst)
    print("warpMatrix:")
    print(m)
    # 基于变换矩阵，将原始图片进行转换
    r = cv2.warpPerspective(img,m,(330,450))
    cv2.imshow("src", img)
    cv2.imshow("result", r)
    cv2.waitKey(0)


