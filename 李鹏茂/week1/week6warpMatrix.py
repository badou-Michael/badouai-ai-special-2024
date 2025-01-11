import numpy as np
import cv2
import matplotlib.pyplot as plt


def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # A*warpMatrix=B
    B = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    A = np.mat(A)
    warpMatrix = A.I * B
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


def main():
    # 读取图片
    image = cv2.imread('photo1.jpg')
    if image is None:
        print("无法读取图片，请检查路径。")
        return

    # 定义源点和目标点
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

    # 计算透视变换矩阵
    warpMatrix = WarpPerspectiveMatrix(src, dst)

    # 应用透视变换
    height, width = image.shape[:2]
    transformed_image = cv2.warpPerspective(image, warpMatrix, (width, height))

    # 展示结果
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title('Transformed Image')
    plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))

    plt.show()


if __name__ == '__main__':
    main()
