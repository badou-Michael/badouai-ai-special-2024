import cv2;


# 使用OpenCV进行最邻近插值
def nearest_inter(image, new_width, new_height):

   resized_image=cv2.resize(image, (new_width, new_height), cv2.INTER_NEAREST)
   return resized_image

# 使用OpenCV进行双线性插值
def linear_inter(image, new_width, new_height):
   resized_image = cv2.resize(image, (new_width, new_height), cv2.INTER_LINEAR)
   return resized_image

img=cv2.imread("../lenna.png") #step1 读取图片
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #将彩色图片灰度化

new_width, new_height = 1000, 1000 #插值新的大小

nearest_image = nearest_inter(img, new_width, new_height) #调用自己的最邻近插值方法
linear_image = linear_inter(img, new_width, new_height) #调用自己的最邻近插值方法
cv2.imshow("nearest inter", nearest_image) #输出最邻近插值后的图像
cv2.imshow("linear inter", nearest_image) #输出双线性插值后的图像
cv2.imshow("image", img) #输出源灰度化的图像
cv2.waitKey(0)
cv2.destroyAllWindows() #关闭所有openCV创建的窗口
