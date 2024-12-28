import cv2
import numpy as np

def perspective_transform(image_path, src_points, dst_points):
    
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_image = cv2.warpPerspective(img, transformation_matrix, (dst_points[3, 0], dst_points[3, 1]))

    return transformed_image


if __name__ == "__main__":
    # Define source and destination points
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

    # Perform perspective transformation
    try:
        result = perspective_transform('photo1.jpg', src, dst)

        # Display images (optional)
        cv2.imshow("Original Image", cv2.imread('photo1.jpg'))
        cv2.imshow("Transformed Image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(e)