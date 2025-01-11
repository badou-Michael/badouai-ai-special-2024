import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

class LinearLeastSquareModel:
    def __init__(self, input_columns, output_columns):
        self.input_columns = input_columns
        self.output_columns = output_columns

    def fit(self, data):
        A = data[:, self.input_columns]
        B = data[:, self.output_columns]
        x, _, _, _ = linalg.lstsq(A, B)
        return x

    def get_error(self, data, model):
        A = data[:, self.input_columns]
        B = data[:, self.output_columns]
        B_fit = np.dot(A, model)
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point

def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None

    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybe_inliers = data[maybe_idxs, :]
        test_points = data[test_idxs, :]
        maybemodel = model.fit(maybe_inliers)
        test_err = model.get_error(test_points, maybemodel)

        also_idxs = test_idxs[test_err < t]
        also_inliers = data[also_idxs, :]

        if len(also_inliers) > d:
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)

            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))

        iterations += 1

    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")

    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit

def random_partition(n, n_data):
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

def test():
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20 * np.random.random((n_samples, n_inputs))
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    B_exact = np.dot(A_exact, perfect_fit)

    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    n_outliers = 100
    all_idxs = np.arange(A_noisy.shape[0])
    np.random.shuffle(all_idxs)
    outlier_idxs = all_idxs[:n_outliers]
    A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
    B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))

    all_data = np.hstack((A_noisy, B_noisy))
    input_columns = range(n_inputs)
    output_columns = [n_inputs + i for i in range(n_outputs)]
    model = LinearLeastSquareModel(input_columns, output_columns)

    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, return_all=True)

    sort_idxs = np.argsort(A_exact[:, 0])
    A_col0_sorted = A_exact[sort_idxs]

    plt.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
    plt.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")
    plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:, 0], label='RANSAC fit')
    plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, perfect_fit)[:, 0], label='exact system')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test()