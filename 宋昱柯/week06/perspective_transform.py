import numpy as np


class PerspectiveTransform:
    """实现透视变换"""

    def __init__(self, src_points: np.ndarray, dst_points: np.ndarray):
        self.src_points = np.asarray(src_points, dtype=np.float64)
        self.dst_points = np.asarray(dst_points, dtype=np.float64)

        assert self.src_points.shape == (4, 2) and self.dst_points.shape == (4, 2)

        self.transform_matrix = self._compute_matrix()

    def _compute_matrix(self) -> np.ndarray:
        """计算透视变换矩阵"""
        x, y = self.src_points[:, 0], self.src_points[:, 1]
        u, v = self.dst_points[:, 0], self.dst_points[:, 1]
        A = np.zeros((8, 8), dtype=np.float64)
        A[::2] = np.column_stack(
            [x, y, np.ones(4), np.zeros(4), np.zeros(4), np.zeros(4), -x * u, -y * u]
        )
        A[1::2] = np.column_stack(
            [np.zeros(4), np.zeros(4), np.zeros(4), x, y, np.ones(4), -x * v, -y * v]
        )
        b = np.zeros(8)
        b[::2] = u
        b[1::2] = v

        # matrix = np.linalg.solve(A,b)
        matrix = np.linalg.inv(A) @ b

        return np.append(matrix, 1.0).reshape(3, 3)

    def __call__(self, src_points: np.ndarray) -> np.ndarray:
        """points.shape=(n,2)"""
        src_points = np.c_[src_points, np.ones(len(src_points))]
        dst_points = src_points @ self.transform_matrix.T
        # 归一化
        return dst_points[:, :2] / dst_points[:, -1, None]


if __name__ == '__main__':
    
    src = np.asarray([[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]])
    dst = np.asarray([[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]])

    transform = PerspectiveTransform(src, dst)
    print(transform.transform_matrix)
