from typing import List
import numpy as np
import sys
from loguru import logger
from pygmo import hypervolume

logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}

def compute_eps_net(points: np.array, num_samples: int = None):
    """Sparsify a set of points returning `num_samples` points that are as
    spread out as possible. Iteratively select the point that is the furthest
    from priorly selected points.
    :param points:
    :param num_samples:
    :return: indices
    """
    assert len(points) > 0, "Need to provide at least 1 point."
    def dist(points, x):
        return np.min([np.linalg.norm(p - x) for p in points])
    n = len(points)
    eps_net = [0]
    indices_remaining = set(range(1,n))
    if num_samples is None:
        num_samples = n
    while len(eps_net) < num_samples and len(indices_remaining) > 0:
        # compute argmin dist(pts[i \not in eps_net], x)
        dist_max = -1
        best_i = 0
        for i in indices_remaining:
            cur_dist = dist(points[eps_net], points[i])
            if cur_dist > dist_max:
                best_i = i
                dist_max = cur_dist
        eps_net.append(best_i)
        indices_remaining.remove(best_i)
    return eps_net


def get_eps_net_ranking(costs, index_list):
    """Produces sorted list containing the best indices
    :param points: Numpy array containing all previous evaluations
    :return: List of num_top indices
    """
    fronts, indices,front_index = nDS_index(np.array([[x[0], x[1]] for x in costs]), index_list)
    logger.debug("fronts:{}",fronts)
    logger.debug("fidxlst:{}",indices)
    logger.debug("front_index:{}",front_index)
    ranked_ids = []
    num_top = len(index_list)
    i = 0
    n_selected = 0
    while n_selected < num_top:
        front = fronts[i]
        front_idx = front_index[i]

        local_order = compute_eps_net(front,
                                      num_samples=(num_top - n_selected))
        logger.debug("local order:{}",local_order)
        ranked_ids += [front_idx[j] for j in local_order]
        i += 1
        n_selected += len(local_order)
    assert len(ranked_ids) == num_top, "Did not assign correct number of \
                                        points to eps-net"

    logger.debug("ranked ids:{}", ranked_ids)
    return ranked_ids


def contributionsHV3D(costs, ref_point= [1, 1]):
    hv = hypervolume(costs)
    return hv.contributions(ref_point)

def minHV3D(costs,ref_point= [1,1]):
    hv = hypervolume(costs)
    return hv.least_contributor(ref_point)


def maxHV3D(costs,ref_point= [1,1]):
    hv = hypervolume(costs)
    return hv.greatest_contributor(ref_point)

def computeHV(costs, ref_point = [1, 1]):
    hv = hypervolume(costs)
    return hv.compute(ref_point)




def pareto_index(costs: np.ndarray, index_list):
    """
    Find the pareto-optimal points
    :param costs: (n_points, m_cost_values) array
    :return: (n_points, 1) indicator if point is on pareto front or not, indices of pareto.
    """
    # first assume all points are pareto optimal
    is_pareto = np.ones(costs.shape[0], dtype=bool)

    for i, c in enumerate(costs):

        if is_pareto[i]:
            # determine all points that have a smaller cost
            all_with_lower_costs = np.any(costs < c, axis=1)
            keep_on_front = np.logical_and(all_with_lower_costs, is_pareto)
            is_pareto = keep_on_front
            is_pareto[i] = True  # keep self

    index_return = index_list[is_pareto]

    return is_pareto, index_return


def nDS_index(costs, index_list):
    """
    Implementation of the non-dominated sorting method
    :param costs: (n_points, m_cost_values) array
    :list of indeces
    :return: list of all fronts, sorted indeces
    """

    dominating_list = []
    index_return_list = []
    fronts = []
    fronts_indices = []
    while costs.size > 0:
        dominating, index_return = pareto_index(costs, index_list)
        fronts.append(costs[dominating])
        fronts_indices.append(index_list[dominating])
        costs = costs[~dominating]
        index_list = index_list[~dominating]
        dominating_list.append(dominating)
        index_return_list.append(index_return)

    return fronts, index_return_list, fronts_indices


def crowdingDist(fronts, index_list):
    """
    Implementation of the crowding distance
    :param front: (n_points, m_cost_values) array
    :return: sorted_front and corresponding distance value of each element in the sorted_front
    """
    dist_list = []
    index_return_list = []

    for g in range(len(fronts)):
        front = fronts[g]
        index_ = index_list[g]

        sorted_front = np.sort(front.view([('', front.dtype)] * front.shape[1]),
                               axis=0).view(np.float)

        _, sorted_index = (list(t) for t in zip(*sorted(zip([f[0] for f in front], index_))))

        normalized_front = np.copy(sorted_front)

        for column in range(normalized_front.shape[1]):
            ma, mi = np.max(normalized_front[:, column]), np.min(normalized_front[:, column])
            normalized_front[:, column] -= mi
            normalized_front[:, column] /= (ma - mi)

        dists = np.empty((sorted_front.shape[0],), dtype=np.float)
        dists[0] = np.inf
        dists[-1] = np.inf

        for elem_idx in range(1, dists.shape[0] - 1):
            dist_left = np.linalg.norm(normalized_front[elem_idx] - normalized_front[elem_idx - 1])
            dist_right = np.linalg.norm(normalized_front[elem_idx + 1] - normalized_front[elem_idx])
            dists[elem_idx] = dist_left + dist_right

        dist_list.append((sorted_front, dists))
        _, index_sorted_max = (list(t) for t in zip(*sorted(zip(dists, sorted_index))))
        index_sorted_max.reverse()

        index_return_list.append(index_sorted_max)

    return dist_list, index_return_list

