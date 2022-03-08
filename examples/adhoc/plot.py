import numpy as np
from pygmo import hypervolume
import os
def maxHV3D(costs):
    hv = hypervolume(costs)
    ref_point = [1,8]
    return hv.greatest_contributor(ref_point)

def pareto(costs: np.ndarray):
    """
    Find the pareto-optimal points
    :param costs: (n_points, m_cost_values) array
    :return: (n_points, 1) indicator if point is on pareto front or not.
    """
    assert type(costs) == np.ndarray
    #assert costs.ndim == 2

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
#cost  = np.loadtxt("C:\\Users\\ayush\\Downloads\\msehvi_1_every_run_cost.txt")
cost  = np.loadtxt("C:\\Users\\ayush\\Downloads\\cost_msehvi_1_every_run_cost.txt")
print("cost:{}",cost)
# print(cost.ndim)
front = pareto(cost)
print(front)
pareto_front= cost[front, :]
print(pareto_front)
from matplotlib import pyplot as plt

plt.scatter(10**cost[:, 0], -cost[:, 1],color='green', marker='o',alpha=0.5,label="sampled_config")
plt.scatter(10**pareto_front[:, 0], -pareto_front[:, 1],color='blue', marker='o',label="pareto")

plt.xlabel('model-param')
plt.ylabel('val_acc')
plt.legend(loc="upper right")
plt.ylim([100, 0])
plt.xscale('log')
plt.show()