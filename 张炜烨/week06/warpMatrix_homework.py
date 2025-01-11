import numpy as np

def compute_perspective_transform(src, dst):

    if src.shape != dst.shape or src.shape[0] < 4 or src.shape[1] != 2:
        raise ValueError("Invalid input shapes.  Both src and dst must be Nx2 arrays with N >= 4.")

    num_points = src.shape[0]
    A = np.zeros((2 * num_points, 8))
    B = dst.reshape(-1, 1)  # Reshape dst to a column vector directly

    for i in range(num_points):
        x, y = src[i]
        x_prime, y_prime = dst[i]

        A[2 * i] = [x, y, 1, 0, 0, 0, -x * x_prime, -y * x_prime]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * y_prime, -y * y_prime]

    try:
        transform_matrix = np.linalg.solve(A, B)  # Use more numerically stable linalg.solve
    except np.linalg.LinAlgError:
        raise ValueError("Singular matrix.  The source points may be collinear.")


    transform_matrix = np.append(transform_matrix, 1)  # Append a_33 = 1
    transform_matrix = transform_matrix.reshape(3, 3)

    return transform_matrix



if __name__ == "__main__":
    src = np.array([[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]])
    dst = np.array([[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]])

    try:
        warp_matrix = compute_perspective_transform(src, dst)
        print("Perspective Transformation Matrix:\n", warp_matrix)
    except ValueError as e:
        print(f"Error: {e}")
