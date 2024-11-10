import cv2
import numpy as np
def bilinear_interpolate(image, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, image.shape[1] - 1)
    x1 = np.clip(x1, 0, image.shape[1] - 1)
    y0 = np.clip(y0, 0, image.shape[0] - 1)
    y1 = np.clip(y1, 0, image.shape[0] - 1)

    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def affine_transformation(image, src, des, new_shape):
    assert src.shape == des.shape and src.shape[0] >= 4

    numPoints = src.shape[0]
    A = np.zeros((numPoints * 2, 8))
    B = np.zeros((numPoints * 2, 1))

    for i in range(0, src.shape[0] * 2):
        # 使用第几组坐标点
        n = i // 2
        if i % 2 == 0:
            # x0 y0 1 0   0 0 -x0X0 -y0X0
            A[i, :] = [src[n, 0], src[n, 1], 1, 0
                    , 0, 0, -src[n, 0] * des[n, 0], -src[n, 1] * des[n, 0]]
            B[i] = des[n, 0]
        else:
            # 0 0 0 x0    y0 1 0 -x0Y0 -y0Y0
            A[i, :] = [0, 0, 0, src[n, 0]
                    , src[n, 1], 1, -src[n, 0] * des[n, 1], -src[n, 1] * des[n, 1]]
            B[i] = des[n, 1]
    A = np.mat(A)
    warpMatrix = A.I * B
    warpMatrix = np.array(warpMatrix)
    warpMatrix = np.append(warpMatrix, 1)
    warpMatrix = warpMatrix.reshape((3, 3))
    warpMatrix = np.mat(warpMatrix)
    print(warpMatrix)

    new_img = np.zeros(new_shape)

    for i in range(new_shape[1]):  # 遍历新图像的列
        for j in range(new_shape[0]):  # 遍历新图像的行
            # 从新图像的左上角到右下角依次插值
            goal = [i, j, 1]  # 目标点坐标(X,Y,Z)
            goal = np.mat(goal)
            goal = goal.reshape((3, 1))
            # print(goal1)
            img_point = warpMatrix.I * goal  # 利用目标点反推原图像点坐标,得到的坐标为(x,y,z),shape=(3,1)
            img_point = img_point.tolist()
            # 其中img_point[0][0]代表原图像x坐标
            # 其中img_point[1][0]代表原图像y坐标
            # 其中img_point[2][0]代表原图像z坐标
            x = int(np.round(img_point[0][0] / img_point[2][0]))  # 计算原图像x坐标
            y = int(np.round(img_point[1][0] / img_point[2][0]))  # 计算原图像y坐标
            # z=int(np.round(img_point[2][0]))
            # print("i=",i,"j=",j,'\n')
            # print("x=", x, 'y=', y, 'z=',z,'\n')
            if y >= image.shape[0]:  # 防止溢出原图像边界
                y = image.shape[0] - 1
            if x >= image.shape[1]:
                x = image.shape[1] - 1
            if y < 0:
                y = 0
            if x < 0:
                x = 0
            new_img[j, i] = image[y, x]

    return new_img

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
src = np.array(src)
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
dst = np.array(dst)
image = cv2.imread("standard/photo1.jpg")
new_image = affine_transformation(image, src, dst, (488, 337, 3))
print(new_image)
cv2.imshow("nima", new_image.astype(np.uint8))
cv2.waitKey(0)