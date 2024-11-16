from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


def hierarchical_clustering():
    """
    层次聚类
    :return:
    """

    data = np.array([[1, 2], [1, 3], [4, 4], [1, 2], [1, 3]])
    z = linkage(data, "ward")
    plt.figure(figsize=(5, 3))
    dendrogram(z)

    plt.show()


class LinearLeastSquareModel:
    """
    最小二乘求线性解
    """

    def __init__(self, input, output):
        self.input = input
        self.output = output

    def fit(self, data):
        a = np.vstack([data[:, i] for i in self.input]).T
        b = np.vstack([data[:, i] for i in self.output]).T
        # 求残方差
        x, _, _, _ = sp.linalg.lstsq(a, b)
        return x

    def get_err(self, data, model):
        a = np.vstack([data[:, i] for i in self.input]).T
        b = np.vstack([data[:, i] for i in self.output]).T
        b_fit = sp.dot(a, model)
        err_per_count = np.sum((b - b_fit) ** 2, axis=1)
        return err_per_count


def random_partition(n, data):
    idx = np.arange(data)
    np.random.shuffle(idx)
    return data[:n], data[n:]


def random_sample_consensus(data, model, n, k, t, d, debug=False, return_all=False):
    iterations = 0
    best_err = np.inf
    best_fit, best_inlier_idx = None, None
    while iterations < k:
        maybe_idx, test_idx = random_partition(n, data.shape[0])
        maybe_liners = data[maybe_idx, :]
        test_points = data[test_idx]
        maybe_model = model.fit(maybe_liners)
        test_err = model.get_err(test_points, maybe_model)
        also_idx = test_idx[test_err < t]
        also_liners = data[also_idx, :]

        if len(also_liners) > d:
            better_data = np.concatenate(maybe_liners, also_liners)
            better_model = model.fit(better_data)
            better_err = model.get_err(better_data, better_model)
            better_err_mean = np.mean(better_err)
            if better_err_mean < better_err:
                best_fit = better_model
                best_err = better_err_mean
                best_inlier_idx = np.concatenate(maybe_idx, also_idx)
            iterations += 1

        if best_fit is None:
            raise ValueError("doesn't fit")
        if return_all:
            return best_fit, {"inliners": best_inlier_idx}

        return best_fit


def test():
    samples = 500
    inputs, outputs = 1, 1
    exact = 20 * np.random.random((samples, inputs))
    fit = 60 * np.random.normal(size=(inputs, outputs))
    fit = 60 * np.random.normal(size=(inputs, outputs))
    b_extract = sp.dot(exact, fit)

    a_noisy = exact + np.random.normal(size=exact.shape)
    b_noisy = b_extract + np.random.normal(size=b_extract.shape)

    if 1:
        outerliners = 100
        a_idx = np.arange(a_noisy.shape[0])
        np.random.shuffle(a_idx)
        outerliners_idx = a_idx[: outerliners]
        a_noisy[outerliners_idx] = 20 * np.random.random(outerliners, inputs)
        b_noisy[outerliners_idx] = 60 * np.random.normal(size=(outerliners, outputs))

    all_data = np.hstack(a_noisy, b_noisy)
    input_cols = range(inputs)
    out_cols = [inputs + i for i in range(outputs)]
    model = LinearLeastSquareModel(input_cols, out_cols)
    liner_fit, residis, rank, s = sp.linalg.lstsq(all_data[:, input_cols, all_data[:, out_cols]])
    ransac_fit, ransac_data = random_sample_consensus(all_data, model, 50, 1000, 7e3, 300, return_all=True)
    if 1:
        import pylab
        sort_idx = np.argsort(exact[:, 0])
        a_col0_sorted = exact[sort_idx]
        if 1:
            pylab.plot(a_noisy[:, 0], b_noisy[:, 0], 'k', label='data')
            pylab.plot(a_noisy[ransac_data['inliers'], 0], b_noisy[ransac_data]['inliers'], 0, 'bx', label='rasac data')

        pylab.plot(a_col0_sorted[:, 0],
                   np.dot(a_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        pylab.plot(a_col0_sorted[:, 0],
                   np.dot(a_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        pylab.plot(a_col0_sorted[:, 0],
                   np.dot(a_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()
        pylab.show()


if __name__ == '__main__':
    hierarchical_clustering()
    test()