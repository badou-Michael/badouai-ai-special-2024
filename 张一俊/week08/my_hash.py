from skimage import util
import numpy as np
import cv2

def get_gaussian_img(image):
    # print(image)
    image_copy = image.copy()
    # gaussian_img = util.noise.random_noise(image_copy, mode='gaussian', var=25)
    # random_noise自动归一化，不需要手动做
    # if image_copy.dtype == np.uint8:
    #     image_copy = image_copy.astype(np.float32) / 255.0
    gaussian_img = util.random_noise(image_copy, mode='gaussian', var=0.01)
    # print(gaussian_img)
    # 将结果转换回 [0, 255] 范围并转换为 uint8 格式
    gaussian_img = (gaussian_img * 255).astype(np.uint8)
    # print(gaussian_img)
    return gaussian_img

def average_hash(image):
    image_copy = image.copy()
    resize_image = cv2.resize(image_copy, (8,8), cv2.INTER_LINEAR)
    # print(resize_image)  # 是整数

    gray_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY) # 是因为gaussian调错接口：Unsupported depth of input image:'VDepth::contains(depth)' where 'depth' is 6 (CV_64F)
    # print(gray_image)
    # print(gray_image.shape)

    sum = 0
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            sum += gray_image[i, j]
    average = int(sum / gray_image.size)
    # print(average)

    aHash = ''
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            if gray_image[i, j] > average:
                aHash += '1'
            else:
                aHash += "0"
    # print(aHash)
    return aHash

def diff_hash(image):
    image_copy = image.copy()
    resize_image = cv2.resize(image_copy, (9, 8), cv2.INTER_LINEAR)
    # print(resize_image.shape)   #(8, 9, 3)

    gray_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)

    dHash = ''
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]-1):
            if gray_image[i, j+1] > gray_image[i, j]:
                dHash += "0"
            else:
                dHash += '1'
    # print(dHash)
    return dHash


def compare_with_hash(a, b):
    if len(a) != len(b):
        print("Error! Can not compare!")
        exit(1)

    s = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            s += 1;
    return s

# if '__main__' == '__name__':
if __name__ == '__main__':
    # gray_image = cv2.imread("lenna.png", flags=0)
    image = cv2.imread("lenna.png")
    noise_image = get_gaussian_img(image)
    # print(noise_image)

    gray_image_aHash = average_hash(image)
    noise_image_aHash = average_hash(noise_image)
    similarity_of_aHash = compare_with_hash(gray_image_aHash, noise_image_aHash)  # 或者叫hash_distance
    # print('\n')
    print(gray_image_aHash)
    print(noise_image_aHash)
    print(similarity_of_aHash)


    print('\n\n')
    gray_image_dHash = diff_hash(image)
    noise_image_dHash = diff_hash(noise_image)
    similarity_of_dHash = compare_with_hash(gray_image_aHash, noise_image_aHash)
    print(gray_image_dHash)
    print(noise_image_dHash)
    print(similarity_of_dHash)
