#coding='utf_8'
import cv2
img = cv2.imread('../../../request/task2/lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('canny',cv2.Canny(gray,160,280))
cv2.waitKey()
cv2.destroyAllWindows()
'''
cv2.destroyAllWindows()函数用于删除所有通过cv2.imshow()创建的窗口。‌
这个函数没有参数，它会删除程序中创建的所有窗口
如果你在程序中创建了多个窗口来显示不同的图像，当你想关闭所有这些窗口时，
可以调用cv2.destroyAllWindows()来实现。这个函数非常有用，尤其是在处理多个窗口并且需要清理资源时。
'''
