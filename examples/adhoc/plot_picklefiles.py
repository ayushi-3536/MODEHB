import numpy as np
from pygmo import hypervolume
from numpy import genfromtxt
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
from ax import Arm
from ax import Metric
from ax import Experiment
from ax import SearchSpace
from ax import SimpleExperiment
from ax import OptimizationConfig

from ax.core.simple_experiment import TEvaluationFunction
from ax.core.simple_experiment import unimplemented_evaluation_function
from baselines import load_experiment

class MultiObjectiveSimpleExperiment(SimpleExperiment):

    def __init__(
        self,
        search_space: SearchSpace,
        optimization_config: OptimizationConfig,
        name: Optional[str] = None,
        eval_function: TEvaluationFunction = unimplemented_evaluation_function,
        status_quo: Optional[Arm] = None,
        properties: Optional[Dict[str, Any]] = None,
        extra_metrics: Optional[List[Metric]] = None,
    ):
        super(MultiObjectiveSimpleExperiment, self).__init__(
            search_space=search_space,
            name=name,
            evaluation_function=eval_function,
            status_quo=status_quo,
            properties=properties
        )

        self.optimization_config = optimization_config

        if extra_metrics is not None:
            for metric in extra_metrics:
                Experiment.add_tracking_metric(self, metric)

# def load_experiment(filename: str):
#     with open(filename, 'rb') as file:
#         return pickle.load(file)

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

def plot_perf_over_time(
    ax: plt.Axes,
    results: np.ndarray,
    times: np.ndarray,
    time_step_size: int,
    is_time_cumulated: bool,
    show: bool = True,
    runtime_upper_bound: Optional[float] = None,
) -> None:
    """
    Args:
        results (np.ndarray):
            The performance of each experiment per evaluation.
            The shape must be (n_experiments, n_evals).
        times (np.ndarray):
            The runtime of each evaluation or cumulated runtime over each experiment.
            The shape must be (n_experiments, n_evals).
        time_step_size (int):
            How many time step size you would like to use for the visualization.
        is_time_cumulated (bool):
            Whether the `times` array already cumulated the runtime or not.
        label (str):
            The name of the plot.
        show (bool):
            Whether showing the plot or not.
            If you would like to pile plots, you need to make it False.
        color (str):
            Color of the plot.
        runtime_upper_bound (Optional[float]):
            The upper bound of runtime to show in the visualization.
            If None, we determine this number by the maximum of the data.
            You should specify this number as much as possible.
    """
    n_experiments, n_evals = results.shape
    results = np.maximum.accumulate(results, axis=-1)

    if not is_time_cumulated:
        times = np.cumsum(np.random.random((n_experiments, n_evals)), axis=-1)
    if runtime_upper_bound is None:
        runtime_upper_bound = times[:, -1].max()

    dt = runtime_upper_bound / time_step_size

    perf_by_time_step = np.full((n_experiments, time_step_size), 1.0)
    curs = np.zeros(n_experiments, dtype=np.int32)

    for it in range(time_step_size):
        cur_time = it * dt
        for i in range(n_experiments):
            while curs[i] < n_evals and times[i][curs[i]] <= cur_time:
                curs[i] += 1
            if curs[i]:
                perf_by_time_step[i][it] = results[i][curs[i] - 1]

    T = np.arange(time_step_size) * dt
    mean = perf_by_time_step.mean(axis=0)
    ste = perf_by_time_step.std(axis=0) / np.sqrt(n_experiments)
    return mean,ste,T


def contributionHV(costs):
    hv = hypervolume(costs)
    #print("hc:{}",hv)
    ref_point = [8,0]
    return hv.compute(ref_point)

def plot_hv(reg_hv,times,ax):


        return   plot_perf_over_time(
            ax,
            reg_hv,
            times,
            time_step_size=500,
            is_time_cumulated=True,

        )



def read_and_generate_hv(cost):
    # cost_bnc = genfromtxt(path, delimiter=',')
    # cost_bnc = cost_bnc[1:, :]
    # print(cost_bnc[:, 2], cost_bnc[:, 1])
    print(cost)
    #cost = np.array([(cost[:, 0]), cost[:, 1]]).T
    print("cost:{}", cost.shape)
    front = pareto(cost)
    print("front:{}",front)
    pareto_front = np.array(cost[front, :])
    print("pareto front:{}",pareto_front)
    hv = [contributionHV(cost[:i+1, :]) for i in range(cost.shape[0])]
    print("hv:{}",hv)
    return hv,pareto_front

import os
import glob

def graph(path,result):
    r = np.array([read_and_generate_hv(np.loadtxt(path + '\\' + f)) for f in result])
    hv_all = r[:, 0]
    pf_all = r[:, 1]
    print("hv all :{}", hv_all)
    print("pf_all shape",pf_all.shape)
    print("pf all :{}", pf_all)
    f_pf=[]
    for i in pf_all:
        for x in i:
            f_pf.append(x)
    cost_pf = np.array(f_pf)
    front_pf = pareto(cost_pf)
    f_pf = np.array(cost_pf[front_pf, :])
    print("final",f_pf)
    print("shape fpf",f_pf.shape)
    # min = np.min(hv_all)
    # shape_all = [len(i) for i in hv_all]
    # print(np.min(shape_all))
    # min = np.min(shape_all)
    # reg_hv = np.array([np.random.choice(i, min) for i in hv_all])
    # times = tuple(range(reg_hv.shape[1]))
    # times = np.tile(times, (10, 1))
    # print("times:{}", times)
    #
    # print("reg_hv shape :{}", reg_hv.shape)
    # plot_hv(reg_hv, times)

    max = np.max(hv_all)
    shape_all = [len(i) for i in hv_all]
    print(np.max(shape_all))
    max = np.max(shape_all)
    print("max :{}", max)
    print("hv plot:{}", len(hv_all[0]))
    temp = np.pad(hv_all[0], (0, max - len(hv_all[0])), mode='maximum')
    print("temp:{}", len(temp))
    reg_hv = np.array([np.pad(i, (0, max - len(i)), mode='maximum') for i in hv_all])
    print("reg hv shape:{}", reg_hv.shape)
    times = tuple(range(reg_hv.shape[1]))
    times = np.tile(times, (10, 1))
    print("times:{}", times)

    print("reg_hv shape :{}", reg_hv.shape)


    return reg_hv,times,f_pf
def read_and_generate_hv_bnc(path):
    cost_bnc = genfromtxt(path, delimiter=',')
    cost_bnc = cost_bnc[1:, :]
    print(cost_bnc[:, 2], cost_bnc[:, 1])
    cost = np.array([np.log10(cost_bnc[:, 2]), -cost_bnc[:, 1]]).T
    print("cost:{}", cost.shape)
    front = pareto(cost)
    pareto_front = np.array(cost[front, :])
    hv = [contributionHV(cost[:i+1, :]) for i in range(cost.shape[0])]
    return hv,pareto_front

# def graph_modehb(path, result):
#         r = [np.loadtxt(path + '\\' + f)[:,0] for f in result]
#         print(len(r))
#         print(r)
#         hv_all = r
#         #hv_all = r[:, 0]
#         #pf_all = r[:, 1]
#         print("hv all :{}", hv_all)
#         #print("pf all :{}", pf_all)
#         # min = np.min(hv_all)
#         shape_all = [len(i) for i in hv_all]
#         print(np.max(shape_all))
#         # min = np.min(shape_all)
#         # reg_hv = np.array([np.random.choice(i, min) for i in hv_all])
#         # times = tuple(range(reg_hv.shape[1]))
#         # times = np.tile(times, (10, 1))
#         # print("times:{}", times)
#         #
#         # print("reg_hv shape :{}", reg_hv.shape)
#         # plot_hv(reg_hv, times)
#
#         #max = np.max(hv_all)
#         shape_all = [len(i) for i in hv_all]
#         print(np.max(shape_all))
#         max = np.max(shape_all)
#         # print("max :{}", max)
#         # print("hv plot:{}", len(hv_all[0]))
#         temp = np.pad(hv_all[0], (0, max - len(hv_all[0])), mode='maximum')
#         print("temp:{}", len(temp))
#         reg_hv = np.array([np.pad(i, (0, max - len(i)), mode='maximum') for i in hv_all])
#         print("reg hv shape:{}", reg_hv.shape)
#         times = tuple(range(reg_hv.shape[1]))
#         times = np.tile(times, (10, 1))
#         print("times:{}", times)
#
#         print("reg_hv shape :{}", reg_hv.shape)
#         return reg_hv, times
    #plot_hv(reg_hv, times)
_, ax = plt.subplots()
path = 'C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\shemoa_15k\\res'
extension = 'txt'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(result)
reg_hv, times, f_pf_shemoa = graph(path,result)
mean,ste,T = plot_hv(reg_hv,times,ax)
ax.plot(T, mean, color="b", label="SHEMOA ")
ax.fill_between(T, mean - ste, mean + ste, color="b", alpha=0.2)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.grid()
#
path = 'C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\msehvi_final\\results'
extension = 'txt'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(result)
reg_hv,times,f_pf_msehvi = graph(path,result)
mean,ste,T = plot_hv(reg_hv,times,ax)
ax.plot(T, mean, color="g", label="MS-EHVI")
ax.fill_between(T, mean - ste, mean + ste, color="g", alpha=0.2)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.grid()
ax.legend(loc="lower right")
path = 'C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\res_modehb\\dehb_run'
extension = 'txt'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(result)
reg_hv,times,f_pf_modehb = graph(path,result)
mean, ste,T = plot_hv(reg_hv,times,ax)
ax.plot(T, mean, color="r", label="MO-DEHB")
ax.fill_between(T, mean - ste, mean + ste, color="r", alpha=0.2)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.grid()
ax.legend(loc="lower right")

################bulk and cut #####################

import os
import glob

path = 'C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\results_save'
extension = 'csv'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
r = np.array([read_and_generate_hv_bnc(path+'\\'+f) for f in result])
hv_all = r[:,0]
pf_all = r[:,1]
print("hv all :{}",hv_all)
print("pf all :{}",pf_all)
f_pf = []
for i in pf_all:
    for x in i:
        f_pf.append(x)
cost = np.array(f_pf)
front = pareto(cost)
f_pf_bnc = np.array(cost[front, :])
print("final", f_pf_bnc)

min = np.min(hv_all)
shape_all = [len(i) for i in hv_all]
print(np.min(shape_all))
min = np.min(shape_all)
reg_hv = np.array([np.random.choice(i, min) for i in hv_all])
times = tuple(range(reg_hv.shape[1]))
times = np.tile(times, (10,1))
print("times:{}", times)

print("reg_hv shape :{}", reg_hv.shape)
#plot_hv(reg_hv,times)

max = np.max(hv_all)
shape_all = [len(i) for i in hv_all]
print(np.max(shape_all))
max = np.max(shape_all)
print("max :{}", max)
print("hv plot:{}",len(hv_all[0]))
temp = np.pad(hv_all[0], (0,max-len(hv_all[0])), mode='maximum')
print("temp:{}",len(temp))
reg_hv = np.array([np.pad(i, (0,max-len(i)), mode='maximum') for i in hv_all])
print("reg hv shape:{}",reg_hv.shape)
times = tuple(range(reg_hv.shape[1]))
times = np.tile(times, (10,1))
print("times:{}", times)

print("reg_hv shape :{}", reg_hv.shape)
mean,ste,T = plot_hv(reg_hv,times,ax)

ax.plot(T, mean, color="m", label="BULK nad Cut")
ax.fill_between(T, mean - ste, mean + ste, color="m", alpha=0.2)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.grid()
ax.legend(loc="lower right")



plt.show()
print("shemoa",f_pf_shemoa)
print("modehb",f_pf_modehb)
print("msehvi",f_pf_msehvi)
plt.scatter(f_pf_shemoa[:, 0],f_pf_shemoa[:, 1], color='g', marker='o',label='SHEMOA',alpha=0.5)
plt.scatter(f_pf_modehb[:,0],f_pf_modehb[:, 1],color='r', marker='x',label='MO-DEHB',alpha=0.5)
plt.scatter(f_pf_msehvi[:, 0],f_pf_msehvi[:, 1],color='b', marker='.',label='MSEHVI',alpha=0.5)
plt.scatter(f_pf_bnc[:, 0],f_pf_bnc[:, 1],color='m', marker='D',label='Bulk and Cut',alpha=0.5)
#plt.xlim(10**2, 10**8)
#plt.ylim(100,0)
plt.title('MODEHB Pareto Front: Fashion dataset')
plt.ylabel('validation-acc')
plt.xlabel('model_param')
plt.legend(loc="upper right")
#plt.xscale('log')
plt.show()


