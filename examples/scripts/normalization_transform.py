from functools import partial
from typing import Optional

import numpy as np
from scipy import stats

import numpy as np

import os
import glob

class temporary_seed:
    def __init__(self, seed):
        self.seed = seed
        self.backup = None

    def __enter__(self):
        self.backup = np.random.randint(2 ** 32 - 1, dtype=np.uint32)
        np.random.seed(self.seed)

    def __exit__(self, *_):
        np.random.seed(self.backup)


class GaussianTransform:
    """
    Transform data into Gaussian by applying psi = Phi^{-1} o F where F is the truncated ECDF.
    :param y: shape (n, dim)
    :param randomize_identical: whether to randomize the rank when consecutive values exists
    if True, draw uniformly inbetween extreme values, if False, use lowest value
    """

    def __init__(self, y: np.array, randomize_identical: bool = True):
        assert y.ndim == 2
        self.dim = y.shape[1]
        self.sorted = y.copy()
        self.sorted.sort(axis=0)
        self.randomize_identical = randomize_identical

    @staticmethod
    def z_transform(series, values_sorted=None, randomize_identical: bool = True):
        # in case of multiple occurences we sample in the interval to get uniform distribution with PIT
        # to obtain deterministic results, we fix the seed locally (and restore the global seed after)
        with temporary_seed(40):
            # applies truncated ECDF then inverse Gaussian CDF.
            if values_sorted is None:
                assert False
                values_sorted = sorted(series)

            def winsorized_delta(n):
                return 1.0 / (4.0 * n ** 0.25 * np.sqrt(np.pi * np.log(n)))

            # delta = winsorized_delta(len(series))
            delta = winsorized_delta(len(values_sorted))

            def quantile(values_sorted, values_to_insert, delta):
                # in case where multiple occurences of the same value exists in sorted array
                # we return a random index in the valid range
                low = np.searchsorted(values_sorted, values_to_insert, side='left')
                if not randomize_identical:
                    res = low
                else:
                    high = np.searchsorted(values_sorted, values_to_insert, side='right')
                    res = np.random.randint(low, np.maximum(high, low + 1))
                return np.clip(res / len(values_sorted), a_min=delta, a_max=1 - delta)

            quantiles = quantile(
                values_sorted,
                series,
                delta
            )

            quantiles = np.clip(quantiles, a_min=delta, a_max=1 - delta)
            # We want 0-1 transforms
            return quantiles
            # return stats.norm.ppf(quantiles)

    def transform(self, y: np.array):
        """
        :param y: shape (n, dim)
        :return: shape (n, dim), distributed along a normal
        """
        assert y.shape[1] == self.dim
        # compute truncated quantile, apply gaussian inv cdf
        return np.stack([
            self.z_transform(y[:, i], self.sorted[:, i], self.randomize_identical)
            for i in range(self.dim)
        ]).T


class StandardTransform:

    def __init__(self, y: np.array):
        assert y.ndim == 2
        self.dim = y.shape[1]
        self.mean = y.mean(axis=0, keepdims=True)
        self.std = y.std(axis=0, keepdims=True)

    def transform(self, y: np.array):
        z = (y - self.mean) / np.clip(self.std, a_min=0.001, a_max=None)
        return z


def from_string(name: str, randomize_identical: bool):
    assert name in ["standard", "gaussian"]
    mapping = {
        "standard": StandardTransform,
        "gaussian": partial(GaussianTransform, randomize_identical=randomize_identical),
    }
    return mapping[name]

if __name__ == '__main__':
    #n = 1000
    tol = 0.05
    dim = 2

    path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\adult\\adult_shemoa'
    extension = 'txt'
    os.chdir(path)
    result = glob.glob('metric*.{}'.format(extension))
    # print(result)
    result = np.array([np.loadtxt(f) for f in result])
    y = np.random.uniform(size=(20, dim))
    print(y)

    print("result")
    print(result[0])
    # GaussianTransform, StandardTransform
    for i,y in enumerate(result):
        print(y.shape)
        psi = GaussianTransform(y)
        z = psi.transform(y)
        print(z)
        print(np.zeros(dim,))
        print(z.mean(axis=0))
        print(np.ones(dim,))
        print(z.std(axis=0))
        f = open(path+'\\norm_{}.txt'.format(i),'w')
        np.savetxt(f,z)
        # assert np.allclose(z.mean(axis=0), np.zeros((dim,)), rtol=tol, atol=tol)
        # assert np.allclose(z.std(axis=0), np.ones((dim,)), rtol=tol)