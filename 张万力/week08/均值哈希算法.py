import cv2
import numpy as np


def average_hash(image, hash_size=8):
    """
    计算图像的均值哈希
    :param image: OpenCV读取的图像
    :param hash_size: 哈希大小，默认8x8
    :return: 哈希值
    """
    # 调整图像大小
    resized = cv2.resize(image, (hash_size, hash_size), interpolation=cv2.INTER_AREA)

    # 转换为灰度
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 计算平均值
    avg = gray.mean()

    # 生成哈希
    hash_value = (gray > avg).astype(int)

    return hash_value


def hamming_distance(hash1, hash2):
    """
    计算两个哈希值的汉明距离
    :param hash1: 第一个哈希值
    :param hash2: 第二个哈希值
    :return: 汉明距离
    """
    # 异或操作
    diff = hash1 ^ hash2

    # 计算1的个数（汉明距离）
    return np.count_nonzero(diff)


def calculate_similarity(hash1, hash2):
    """
    计算哈希相似度
    :param hash1: 第一个哈希值
    :param hash2: 第二个哈希值
    :return: 相似度百分比
    """
    # 哈希值大小
    hash_size = 8

    # 计算汉明距离
    distance = hamming_distance(hash1, hash2)

    # 计算相似度
    # 总位数为hash_size的平方,缩放为8*8
    max_distance = hash_size * hash_size

    # 计算相似度百分比
    similarity = (1 - distance / max_distance) * 100

    return similarity


def compare_images(image1_path, image2_path):
    """
    比较两张图片的相似度
    :param image1_path: 第一张图片路径
    :param image2_path: 第二张图片路径
    :return: 相似度
    """
    # 读取图片
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # 检查图片是否成功读取
    if img1 is None or img2 is None:
        print("无法读取图片，请检查路径是否正确")
        return None

    # 计算哈希值
    hash1 = average_hash(img1)
    hash2 = average_hash(img2)

    # 计算相似度
    similarity = calculate_similarity(hash1, hash2)

    return similarity


# 使用示例
def main():
    # 替换为你的图片路径
    image1_path = 'iphone1.png'
    image2_path = 'iphone2.png'

    # 比较图片
    similarity = compare_images(image1_path, image2_path)

    if similarity is not None:
        print(f"图片相似度: {similarity:.2f}%")


if __name__ == "__main__":
    main()
