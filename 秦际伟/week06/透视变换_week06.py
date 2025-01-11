import cv2
import numpy as np

def apply_perspective_transform(image_path, src, dst, output_size):
    """
    应用透视变换。

    参数:
    - image_path: 图像文件路径
    - src: 源图像的四个顶点坐标 (4x2 的 numpy 数组)
    - dst: 目标图像的四个顶点坐标 (4x2 的 numpy 数组)
    - output_size: 输出图像的尺寸 (宽度, 高度)

    返回:
    - img: 原始图像
    - result: 变换后的图像
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("图片读取失败")
        
        assert src.shape == (4, 2), "源点必须是4个二维坐标"
        assert dst.shape == (4, 2), "目标点必须是4个二维坐标"
        
        result3 = img.copy()
        m = cv2.getPerspectiveTransform(src, dst)
        result = cv2.warpPerspective(result3, m, output_size)
        
        return img, result
    except Exception as e:
        print(f"错误: {e}")
        return None, None

if __name__ == "__main__":
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    img, result = apply_perspective_transform('photo1.jpg', src, dst, (337, 488))
    
    if img is not None and result is not None:
        cv2.imshow("src", img)
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
