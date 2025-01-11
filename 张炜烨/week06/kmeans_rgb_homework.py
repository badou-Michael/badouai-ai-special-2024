import cv2
import numpy as np
import matplotlib.pyplot as plt

def quantize_image(image, k_values):
    data = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    quantized_images = []

    for k in k_values:
        compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)
        centers = np.uint8(centers)
        quantized_image = centers[labels.flatten()].reshape(image.shape)
        quantized_images.append(quantized_image)

    return quantized_images

if __name__ == "__main__":
    img = cv2.imread('lenna.png')
    if img is None:
        print("Error: Could not read image 'lenna.png'")
        exit()

    k_values = [2, 4, 8, 16, 64]
    quantized_images = quantize_image(img, k_values)

    # Display the images using Matplotlib
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("K-Means Image Quantization", fontsize=16)

    images = [img] + quantized_images
    titles = ["Original Image"] + [f"K = {k}" for k in k_values]

    for ax, image, title in zip(axes.flatten(), images, titles):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to prevent title overlap
    plt.show()