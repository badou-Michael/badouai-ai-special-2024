import cv2

'''
Canny边缘检测：动态的参数调整
'''


def callback_CannyThreshold(cur_threshold):
    print(cur_threshold)
    gray_canny = cv2.Canny(gray, cur_threshold, cur_threshold * 3, apertureSize=3)
    dst = cv2.bitwise_and(img, img, mask=gray_canny)
    cv2.imshow('CANNY_adjust', dst)
    cv2.imshow('CANNY_adjust_2', gray_canny)  # 窗口名字相同就会显示在一个窗口，不同就会出现多个窗口


img = cv2.imread('lenna.png')
gray = cv2.imread('lenna.png', 0)
cv2.namedWindow('CANNY_adjust')  # 窗口名字相同就会显示在一个窗口，不同就会出现多个窗口
cv2.createTrackbar('params', 'CANNY_adjust', 0, 100, callback_CannyThreshold)  # 65，回调函数自带参数
callback_CannyThreshold(0)  # 初始显示!!!
if cv2.waitKey(0) == 27:
    # Esc键
    cv2.destroyAllWindows()
