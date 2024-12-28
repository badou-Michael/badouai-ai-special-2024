import numpy as np
import random

class LinearLeastSquareModel:
    def __init__(self):
        self.slope = 0
        self.intercept = 0
    
    def fit(self, data):
        if len(data) < 2:
            raise ValueError("Not enough points to fit a line")
        x, y = data[:, 0], data[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        self.slope, self.intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    
    def get_error(self, data):
        x, y = data[:, 0], data[:, 1]
        y_pred = self.slope * x + self.intercept
        return np.abs(y - y_pred)

def ransac(data, model, n, k, t, d):
    best_model = None
    best_inliers = None
    best_inliers_count = 0

    while k > 0:
        k -= 1
        maybe_inliers = data[random.sample(range(data.shape[0]), n)]
        model.fit(maybe_inliers)
        errors = model.get_error(data)
        inliers = data[errors < t]
        if len(inliers) > d and len(inliers) > best_inliers_count:
            best_model = LinearLeastSquareModel()
            best_model.fit(inliers)
            best_inliers = inliers
            best_inliers_count = len(inliers)
    
    return best_model, best_inliers

if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(0)
    n_points = 100
    X = np.linspace(0, 10, n_points)
    Y = 2 * X + 1 + np.random.normal(size=n_points)  # Line with some noise
    outliers = np.array([np.random.randint(0, 10, 10), np.random.randint(0, 30, 10)]).T
    data = np.vstack((np.column_stack((X, Y)), outliers))  # Combine data with outliers

    # Define parameters
    n = 2           # Minimum number of points to estimate model
    k = 100         # Maximum number of iterations
    t = 3           # Threshold for inliers
    d = n_points    # Minimum number of inliers required

    # Run RANSAC
    model = LinearLeastSquareModel()
    best_model, best_inliers = ransac(data, model, n, k, t, d)

    # Output results
    print("Best model parameters:")
    if best_model:
        print("Slope:", best_model.slope)
        print("Intercept:", best_model.intercept)
        print("Number of inliers:", len(best_inliers))
    else:
        print("No valid model found.")
