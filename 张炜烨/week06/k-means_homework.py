import cv2
import numpy as np
import matplotlib.pyplot as plt

def kmeans_segmentation(image, k=4, max_iter=10, epsilon=1.0, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS):
    
    if image is None:
        raise ValueError("Input image cannot be None")
    if len(image.shape) > 2 and image.shape[2] > 1:
        raise ValueError("Input image must be grayscale.")


    data = image.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    _, labels, _ = cv2.kmeans(data, k, None, criteria, attempts, flags)
    segmented_image = labels.reshape(image.shape).astype(np.uint8)
    return segmented_image


if __name__ == "__main__":
    img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not read image 'lenna.png'")
        exit()



    segmented_img = kmeans_segmentation(img)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("K-Means Image Segmentation", fontsize=16)

    images = [img, segmented_img]
    titles = ["Original Image", "Segmented Image"]

    for ax, image, title in zip(axes.flatten(), images, titles):
        ax.imshow(image, cmap='gray')  # Use cmap='gray' for grayscale images
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()