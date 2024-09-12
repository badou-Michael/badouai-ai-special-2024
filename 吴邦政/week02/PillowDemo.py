from PIL import Image


#打开图片
image = Image.open("image.jpg")
#通过pillow库 convert方法转化为灰度图 参数L为 整数灰度图
gray_image = image.convert("L")
#展示
gray_image.show()

#二值图阈值
threshold = 128
#转化为二值图 p代表每个像素的值
binary_image = gray_image.point(lambda p: 255 if p > threshold else 0)
#展示
binary_image.show()
