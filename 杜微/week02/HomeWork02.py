from PIL import Image

# 实现图片的灰度图转换和二值图转换

image = Image.open(r'b_(61) 拷贝 3.jpg')
# image_convert = image.convert('L')

# 图片灰度化
# 获取原图大小
image_size = image.size
image_load = image.load()
# 创建新的空灰度图
image_new = Image.new('L', image_size)
# 获取新元素图数据
new_load = image_new.load()
# 便利原图的每一个元素点
for i in range(image.width):
    for j in range(image.height):
       r,g,b = image_load[i, j]
       # 根据原图的每一像素点的信息，计算出对应亮度元素点的数值
       # 将计算出的灰度元素值存放新的灰度图中
       new_load[i, j] = int(0.299 * r + 0.587 * g + 0.114 * b)

image_new.show()


image_point2 = image_new.point(lambda x: 0 if x < 128 else 255)

image_point2.show()
