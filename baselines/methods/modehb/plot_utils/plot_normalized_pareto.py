from functools import partial
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

import numpy as np

def pareto(costs: np.ndarray):
    """
    Find the pareto-optimal points
    :param costs: (n_points, m_cost_values) array
    :return: (n_points, 1) indicator if point is on pareto front or not.
    """
    assert type(costs) == np.ndarray
    assert costs.ndim == 2

    # first assume all points are pareto optimal
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            # determine all points that have a smaller cost
            all_with_lower_costs = np.any(costs < c, axis=1)
            keep_on_front = np.logical_and(all_with_lower_costs, is_pareto)
            is_pareto = keep_on_front
            is_pareto[i] = True  # keep self
    return is_pareto

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
    n = 1000
    tol = 0.05
    dim = 2
    y = np.random.uniform(size=(n, dim))
    #print(y)


    #cost = np.loadtxt("/content/drive/MyDrive/run/changedk/imagenet16/32_runtime_b199/_1/every_run_cost_1634500997.1196012.txt")
    cost = np.loadtxt("/content/drive/MyDrive/run/imagenet16/optimalhvimprov20011/optimal_metric_pertime.txt")
    #print(cost)
    y=np.array(cost)
    print("cost shape:{}, n dim:{}",y.shape,y.ndim)
    # GaussianTransform, StandardTransform
    psi = GaussianTransform(y)
    print(psi)
    #print("cost transform shape:{}",psi.shape)
    z = psi.transform(y)
    front = pareto(z)
    print(front)
    zp= z[front, :]
    print("z is done for cost")
    #zp = np.loadtxt("/content/drive/MyDrive/run/imagenet16/optimalhvimprov20011/pareto_pertime.txt")
    # print("pareto shape:",p1.shape)
    # print("pareto array shape:",np.array(p1).shape)
    # print("cost:{}, pareto:{}",cost.ndim,p1.ndim)
    # psi_p = GaussianTransform(p1)
    # #print("p dim:{}",psi.ndim)
    # #print("psi shape:{}",psi_p.ndim)
    # zp = psi_p.transform(p1)
    print(zp)
    plt.scatter(z[:, 0], z[:, 1],color='green', marker='o',label="sampled_config")
    plt.scatter(zp[:, 0], zp[:, 1],color='blue', marker='o',label="pareto")

    plt.title('MODEHB Pareto Front: Cifar-10, runtime=24h, hp=200')
    plt.xlabel('validation-acc')
    plt.ylabel('model_param')
    plt.legend(loc="upper right")
    plt.show()
