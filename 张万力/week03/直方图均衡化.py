import cv2

# 对图像执行直方图均衡化，is_color 表示图像是否为彩色
def histogram_equalization(img, is_color):
    if is_color:
        # 彩色图像：转换到YCrCb颜色空间并对亮度通道（ycrcb_img[:, :, 0]）进行直方图均衡化,保持颜色的自然。Cr=ycrcb_img[:, :, 1] Cb=ycrcb_img[:, :, 2]
        ycrcb_img=cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)
        ycrcb_img[:, :, 0]=cv2.equalizeHist(ycrcb_img[:, :, 0])
        equalized_img = cv2.cvtColor(ycrcb_img,cv2.COLOR_YCrCb2BGR)
    else:
        # 灰色图像：进行直方图均衡化
        equalized_img = cv2.equalizeHist(img)
    return equalized_img

# 对图像执行所有通道均衡化
def histogram_equalization_color(img):
    # 分离颜色通道
    channels=cv2.split(img)

    # 对每个通道进行直方图均衡化
    all_channels = [cv2.equalizeHist(channel) for channel in channels]

    #合并通道
    equalized_img=cv2.merge(all_channels)
    return equalized_img


img = cv2.imread("../lenna.png")  # step1 读取图片
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将彩色图片灰度化

# 直方图均衡化
equalized_color = histogram_equalization(img, True)
equalized_gray= histogram_equalization(gray_img, False)
equalized_image = histogram_equalization_color(img)


cv2.imshow("equalized color", equalized_color) #输出亮度通道直方图均衡化
cv2.imshow("equalized gray", equalized_gray) #输出灰度直方图均衡化
cv2.imshow("equalized image",equalized_image) #输出所有通道直方图均衡化
cv2.imshow(" image",img) #输出原图
cv2.waitKey(0)
cv2.destroyAllWindows() #关闭所有openCV创建的窗口
