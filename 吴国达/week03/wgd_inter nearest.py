# 最邻近插值实现图像缩放
import cv2
import numpy as np
import win32gui


def cv_set_title(old_title, new_title='中文', one_run=False):
    """
    设置窗口标题
    :param old_title: 旧标题
    :param new_title: 新标题
    :param one_run: 是否只运行一次
    :return:
    """
    if not one_run:
        # 根据窗口名称查找其句柄 然后使用函数修改其标题
        # 尽量选择一个不常见的英文名 防止误该已有#的窗口标题 初始化时通常取无意义的名字  比如这里取‘aa’
        handle = win32gui.FindWindow(0, old_title)
        win32gui.SetWindowText(handle, new_title)
        one_run = True
    return one_run


# 缩放图像
def img_resize(image, des_h, des_w):
    height, width, channels = image.shape
    des_image = np.zeros((des_h, des_w, channels), np.uint8)
    sh = des_h / height
    sw = des_w / width
    for i in range(des_h):
        for j in range(des_w):
            x = int(i / sh + 0.5)  # 四舍五入取邻近值
            y = int(j / sw + 0.5)
            des_image[i, j] = image[x, y]
    return des_image


img = cv2.imread("images/lenna.png")
cv2.imshow("src image", img)
cv_set_title('src image', new_title='原图')
new_img = img_resize(img, 800, 800)
cv2.imshow("inter nearest", new_img)
cv_set_title('inter nearest', new_title='最邻近插值')
cv_resize = cv2.resize(img, (800, 800), interpolation=cv2.INTER_NEAREST)
cv2.imshow("cv_resize", cv_resize)

cv2.waitKey(0)
