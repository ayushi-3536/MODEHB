#%matplotlib inline
import numpy as np
from pygmo import hypervolume
from numpy import genfromtxt
from typing import Optional
import matplotlib.pyplot as plt
from baselines import load_experiment

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

def plot_perf_over_time(
    ax: plt.Axes,
    results: np.ndarray,
    times: np.ndarray,
    time_step_size: int,
    is_time_cumulated: bool,
    label: str,
    show: bool = True,
    color: str = "red",
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
    ax.plot(T, mean, color=color, label=label)
    ax.fill_between(T, mean - ste, mean + ste, color=color, alpha=0.2)
    #ax.set_xscale("log")
    #ax.set_yscale("log")
    ax.grid()
    ax.legend(loc="lower right")

def contributionHV(costs):
    hv = hypervolume(costs)
    #print("hc:{}",hv)
    ref_point = [8,0]
    return hv.compute(ref_point)

def plot_hv(reg_hv,times):
    _, ax = plt.subplots()
    for idx, (col, hpo) in enumerate(
            zip(["red"], ["MO-DEHB"])
    ):
        plot_perf_over_time(
            ax,
            reg_hv,
            times,
            time_step_size=200,
            is_time_cumulated=False,
            color=col,
            label=hpo,
        )
    plt.show()


def read_and_generate_hv(path):
    cost_bnc = genfromtxt(path, delimiter=',')
    cost_bnc = cost_bnc[1:, :]
    print(cost_bnc[:, 2], cost_bnc[:, 1])
    cost = np.array([np.log10(cost_bnc[:, 2]), -cost_bnc[:, 1]]).T
    print("cost:{}", cost.shape)
    front = pareto(cost)
    pareto_front = np.array(cost[front, :])
    hv = [contributionHV(cost[:i+1, :]) for i in range(cost.shape[0])]
    return hv,pareto_front
import timeit
import _pickle as cPickle
import gc


def load_cpickle_gc(path):
    output = open(path, 'rb')

    # disable garbage collector
    gc.disable()

    mydict = cPickle.load(output)

    # enable garbage collector again
    gc.enable()
    output.close()
    return mydict




from tqdm import tqdm
import pickle
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.pareto_frontier import plot_pareto_frontier
import pandas as pd
# open a file, where you stored the pickled data
def process_picklefiles(path,save_path,i):
    #file = open(path, 'rb')
    # dump information to that file
    data = load_experiment(path)
    #print("data",data)
    #print("data df",data.df)
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #        print(data.df)
    #print("data wallclock:",(experiment.trials[i - 1].time_completed - experiment.time_created).total_seconds())
    d = data.fetch_data()
    df = vars(d)
    df = df['_df']
    #df = data.df
    #pd.set_option('max_colwidth', None)

    print("df",df)

    df = df.drop(columns=['sem'])

    df_acc3 = df[(df['metric_name']=='val_acc_3')]
    print(df_acc3)
    df_acc3.set_index('trial_index')
    df_acc = df[(df['metric_name']=='val_acc_1')]
    df_acc.set_index('trial_index')
    df_np = df = df[(df['metric_name']=='num_params')]
    df_np.set_index('trial_index')
    #df_tr = df = df[(df['metric_name']=='total_runtime')]
    #df_tr.set_index('trial_index')
    #df_er = df = df[(df['metric_name']=='eval_runtime')]
    #df_er.set_index('trial_index')
    #cost = df_acc['mean']
    print("acc:",df_acc)
    df_np = df_np.drop(columns=['metric_name','arm_name'])
    df_acc = df_acc.drop(columns=['metric_name','arm_name'])
    df_acc3 = df_acc3.drop(columns=['metric_name','arm_name'])
    #df_tr = df_tr.drop(columns=['metric_name','arm_name'])
    #df_er = df_er.drop(columns=['metric_name','arm_name'])
    print("acc:{}",df_acc)
    print("df_scc3:{}",df_acc3)
    #print(df_np)

    bd = pd.merge(df_np, df_acc, on=['trial_index'])
    #bd = pd.merge(bd, df_tr, on=['trial_index'])
    #bd = pd.merge(bd, df_er, on=['trial_index'])
    bd = bd.drop(columns=['trial_index'])

    bd = bd.to_numpy()
    #print(bd)
    print(bd.ndim)
    cost=bd
    front = pareto(cost)
    pareto_front= cost[front, :]
    print(pareto_front)
    p1= pareto_front

    #bd3 = pd.merge(df_np, df_acc,on=['trial_index'])
    #bd3 = bd3.drop(columns=['trial_index'])

    #bd3 = bd3.to_numpy()
    #print(bd)
    #print("cost:{}",cost)
    #cost3=bd3
    file_name = open(save_path + '_'+str(i)+'_all_cost.txt', 'w')
                    # for line in costs:
    np.savetxt(file_name, cost)
    result = {
            'hypervolume': [0.0],
            'walltime': [0.0],
            'evaluations': [0.0]
    }



    for i in tqdm(range(1, data.shape[0]), desc=experiment.name):
        result['hypervolume'].append(hv.compute(data[:i][is_non_dominated(data[:i])]))
        result['walltime'].append(((experiment.trials[i - 1].time_completed - experiment.time_created).total_seconds()))
        result['evaluations'].append(i)
    print("data wallclock:",result)
    #file_name = open(save_path + '_'+str(i)+'_cost_moshb_1_every_run_cost.txt', 'w')
                    # for line in costs:
    #np.savetxt(file_name, cost)

import os
import glob
print("path:{}",os.getcwd())
path = os.getcwd()+'/shemoa_15k'
extension = 'pickle'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(result)
for i, f in enumerate(result):
    print("processing",path+'/'+f)
    process_picklefiles(path+'/'+f, path,i)
    break
# r = np.array([read_and_generate_hv(path+'\\'+f) for f in result])
# hv_all = r[:,0]
# pf_all = r[:,1]
# print("hv all :{}",hv_all)
# print("pf all :{}",pf_all)
# min = np.min(hv_all)
# shape_all = [len(i) for i in hv_all]
# print(np.min(shape_all))
# min = np.min(shape_all)
# reg_hv = np.array([np.random.choice(i, min) for i in hv_all])
# times = tuple(range(reg_hv.shape[1]))
# times = np.tile(times, (10,1))
# print("times:{}", times)
#
# print("reg_hv shape :{}", reg_hv.shape)
# plot_hv(reg_hv,times)
#
# max = np.max(hv_all)
# shape_all = [len(i) for i in hv_all]
# print(np.max(shape_all))
# max = np.max(shape_all)
# print("max :{}", max)
# print("hv plot:{}",len(hv_all[0]))
# temp = np.pad(hv_all[0], (0,max-len(hv_all[0])), mode='maximum')
# print("temp:{}",len(temp))
# reg_hv = np.array([np.pad(i, (0,max-len(i)), mode='maximum') for i in hv_all])
# print("reg hv shape:{}",reg_hv.shape)
# times = tuple(range(reg_hv.shape[1]))
# times = np.tile(times, (10,1))
# print("times:{}", times)
#
# print("reg_hv shape :{}", reg_hv.shape)
# plot_hv(reg_hv,times)

# plt.plot(i, color='green', marker='o',alpha=0.5)
#     # plt.plot(hv1,color='red', marker='o',alpha=0.5)
#     # plt.plot(hv2,color='blue', marker='o',alpha=0.5)
# plt.title('HyperVolume: FashionNet: bulkandcut')
# plt.xlabel('epoch')
# plt.xscale('log')
# plt.legend(loc="lower right")
# plt.ylabel('hypervolume')
# plt.show()
#
#pareto_front = pareto_front[:,0],pareto_front[:,1]]
#print("pareto:{}",pareto_front)
# print("cost bnc:{}", cost_bnc)
# print("pareto_front:{}", pareto_front)
# plt.scatter(np.log10(cost[:,2]),-cost[:,1],color='green', marker='o',alpha=0.5)
# plt.scatter(pareto_front[:,0],pareto_front[:,1],color='red', marker='o',alpha=0.5)

# cost = np.loadtxt("/content/multi-obj-baselines/results_save/population_summary_1.csv")
# plt.plot(cost[:,0]/3600,cost[:,1],color='green', marker='o',alpha=0.5,label='24h, seed=8')
# cost1 = np.loadtxt("/content/MODEHB/fashion_runs/fashion_logs_24h_7/hv_contribution.txt")
# plt.plot(cost1[:,0]/3600,cost1[:,1],color='blue', marker='x',alpha=0.5,label='24h, seed=7')
# cost2 = np.loadtxt("/content/MODEHB/fashion_runs/fashion_logs_24h_10/hv_contribution.txt")
# plt.plot(cost2[:,0]/3600,cost2[:,1],color='red', marker='.',alpha=0.3,label='24h, seed=10')
# plt.title('Sample Configuration: FashionNet: bulkandcut')
# plt.xlabel('n_param')
# plt.xscale('log')
# plt.legend(loc="lower right")
# plt.ylabel('accuracy')
#
# plt.show()

