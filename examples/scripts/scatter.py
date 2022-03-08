import glob
from copy import deepcopy
from typing import List, Optional, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from baselines import load_experiment
#from comparison import load_experiment

plt.style.use('ggplot')

import torch
from ax import Experiment, MultiObjectiveOptimizationConfig
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from matplotlib.cm import ScalarMappable

from collections import defaultdict

from tqdm import tqdm


def plot_aggregated_scatter(
        path: str,
        experiments_names: List[str],
        metric_x: Optional[str] = None,
        metric_y: Optional[str] = None,
        bound_max_x: Optional[float] = None,
        bound_max_y: Optional[float] = None,
):

    experiments = [
        [
            load_experiment(e) for e in glob.glob(f'{path}/{exp_name}_*_final.pickle')
        ] for exp_name in experiments_names
    ]

    metrics = experiments[0][0].optimization_config.objective.metrics

    metric_x = metrics[0] if metric_x is None else next(m for m in metrics if m.name == metric_x)
    metric_y = metrics[1] if metric_y is None else next(m for m in metrics if m.name == metric_y)

    fig, axes = plt.subplots(1, len(experiments), figsize=(5 * len(experiments), len(experiments)))

    for ax, experiment_list in zip(axes, experiments):

        for i, experiment in enumerate(experiment_list[:5]):

            data = experiment.fetch_data().df
            values_x = data[data['metric_name'] == metric_x.name]['mean'].values
            values_y = data[data['metric_name'] == metric_y.name]['mean'].values

            values_x = values_x if metric_x.lower_is_better else -1 * values_x
            values_y = values_y if metric_y.lower_is_better else -1 * values_y

            num_elements = values_x.size

            iterations = np.arange(num_elements)

            if i == 0:
                ax.set_title(experiment.name.split('_')[0])
                ax.set_xlabel(metric_x.name)
                ax.set_ylabel(metric_y.name)

            ax.scatter(values_x, values_y, c=iterations, alpha=0.8)

            th_list = experiment.optimization_config.objective_thresholds
            min_x = next(th.bound for th in th_list if th.metric.name == metric_x.name)
            min_y = next(th.bound for th in th_list if th.metric.name == metric_y.name)

            ax.set_xlim(right=8)
            if bound_max_x or True:
                ax.set_xlim(left=2)

            ax.set_ylim(top=0)
            if bound_max_x or True:
                ax.set_ylim(bottom=-110)

            if i == 0:
                norm = plt.Normalize(iterations.min(), iterations.max())
                sm = ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('viridis'))

                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.15)
                fig.colorbar(sm, cax=cax, label='Iterations')

    plt.savefig(f'{path}/evals_overtime.pdf', dpi=450)


def plot_aggregated_pareto_fronts(
        path: str,
        experiments_names: List[str],
        metric_x: Optional[str] = None,
        metric_y: Optional[str] = None,
        bound_max_x: Optional[float] = None,
        bound_max_y: Optional[float] = None,
):
    experiments = [
        [
            e for e in glob.glob(f'{path}/{exp_name}_*_final.pickle')
        ] for exp_name in experiments_names
    ]


    sample_experiment = load_experiment(experiments[0][0])
    metrics = sample_experiment.optimization_config.objective.metrics

    metric_x = metrics[0] if metric_x is None else next(m for m in metrics if m.name == metric_x)
    metric_y = metrics[1] if metric_y is None else next(m for m in metrics if m.name == metric_y)

    fig, ax = plt.subplots()

    th_list = sample_experiment.optimization_config.objective_thresholds
    #hv = Hypervolume(torch.tensor([(-1 if th.metric.lower_is_better else 1) * th.bound for th in th_list][::-1], device=torch.device("cuda:0")))
    hv = Hypervolume(torch.tensor([(-1 if th.metric.lower_is_better else 1) * th.bound for th in th_list][::-1], device=torch.device("cpu")))
    for exp in experiments:
        if len(exp) == 0:
            continue

        #print('Doing: ', exp)

        exp = load_experiment(exp[0])


        data = exp.fetch_data().df
        values_x = data[data['metric_name'] == metric_x.name]['mean'].values
        values_y = data[data['metric_name'] == metric_y.name]['mean'].values

        maximizator = [-1 if metric_x.lower_is_better else 1, -1 if metric_y.lower_is_better else 1]

        values = np.asarray([values_x, values_y]).T
        values = values[is_non_dominated(torch.as_tensor(values * maximizator))]
        values = values[values[:, 0].argsort()]

        #total_hv = int(hv.compute(torch.tensor(values * maximizator, device=torch.device("cuda:0"))))
        total_hv = int(hv.compute(torch.tensor(values * maximizator, device=torch.device("cpu"))))
        ax.plot(values[:, 0], values[:, 1], '-o', lw=3.0, label=exp.name + ' ' + str(total_hv))

    ax.set_xlabel(metric_x.name)
    ax.set_ylabel(metric_y.name)

    th_list = sample_experiment.optimization_config.objective_thresholds
    min_x = next(th.bound for th in th_list if th.metric.name == metric_x.name)
    min_y = next(th.bound for th in th_list if th.metric.name == metric_y.name)

    ax.set_xlim(right=min_x)
    if bound_max_x:
        ax.set_xlim(left=bound_max_x)

    ax.set_ylim(top=min_y)
    if bound_max_x:
        ax.set_ylim(bottom=bound_max_y)

    ax.legend()

    plt.savefig(f'{path}/paretofronts.pdf', dpi=450)


def plot_pareto_fronts(
        experiments: List[Experiment],
        metric_x: Optional[str] = None,
        metric_y: Optional[str] = None,
        bound_max_x: Optional[float] = None,
        bound_max_y: Optional[float] = None,
):

    metrics = experiments[0].optimization_config.objective.metrics

    metric_x = metrics[0] if metric_x is None else next(m for m in metrics if m.name == metric_x)
    metric_y = metrics[1] if metric_y is None else next(m for m in metrics if m.name == metric_y)

    fig, ax = plt.subplots()

    th_list = experiments[0].optimization_config.objective_thresholds
    hv = Hypervolume(torch.tensor([(-1 if th.metric.lower_is_better else 1) * th.bound for th in th_list], device=torch.device("cuda:0")))

    for exp in experiments:
        data = exp.fetch_data().df
        values_x = data[data['metric_name'] == metric_x.name]['mean'].values
        values_y = data[data['metric_name'] == metric_y.name]['mean'].values

        maximizator = [-1 if metric_x.lower_is_better else 1, -1 if metric_y.lower_is_better else 1]

        values = np.asarray([values_x, values_y]).T
        values = values[is_non_dominated(torch.as_tensor(values * maximizator))]
        values = values[values[:, 0].argsort()]

        total_hv = hv.compute(torch.tensor(values * maximizator, device=torch.device("cuda:0")))

        ax.plot(values[:, 0], values[:, 1], '-o', lw=3.0, label=exp.name + ' ' + str(total_hv))

    ax.set_xlabel(metric_x.name)
    ax.set_ylabel(metric_y.name)

    th_list = experiments[0].optimization_config.objective_thresholds
    min_x = next(th.bound for th in th_list if th.metric.name == metric_x.name)
    min_y = next(th.bound for th in th_list if th.metric.name == metric_y.name)

    ax.set_xlim(right=min_x)
    if bound_max_x:
        ax.set_xlim(left=bound_max_x)

    ax.set_ylim(top=min_y)
    if bound_max_x:
        ax.set_ylim(bottom=bound_max_y)

    ax.legend()

    plt.savefig('figures/paretofronts2.pdf', dpi=450)

#
def plot_scatter(
        experiments: List[Experiment],
        metric_x: Optional[str] = None,
        metric_y: Optional[str] = None,
        bound_max_x: Optional[float] = None,
        bound_max_y: Optional[float] = None,
):
    metrics = experiments[0].optimization_config.objective.metrics

    metric_x = metrics[0] if metric_x is None else next(m for m in metrics if m.name == metric_x)
    metric_y = metrics[1] if metric_y is None else next(m for m in metrics if m.name == metric_y)

    fig, axes = plt.subplots(len(experiments), figsize=(5, 5 * len(experiments)))

    for ax, experiment in zip(axes, experiments):

        data = experiment.fetch_data().df
        values_x = data[data['metric_name'] == metric_x.name]['mean'].values
        values_y = data[data['metric_name'] == metric_y.name]['mean'].values

        values_x = values_x if metric_x.lower_is_better else -1 * values_x
        values_y = values_y if metric_y.lower_is_better else -1 * values_y

        num_elements = values_x.size

        iterations = np.arange(num_elements)

        ax.set_title(experiment.name)
        ax.scatter(values_x, values_y, c=iterations, alpha=0.8)

        ax.set_xlabel(metric_x.name)
        ax.set_ylabel(metric_y.name)

        th_list = experiment.optimization_config.objective_thresholds
        min_x = next(th.bound for th in th_list if th.metric.name == metric_x.name)
        min_y = next(th.bound for th in th_list if th.metric.name == metric_y.name)

        ax.set_xlim(right=min_x)
        if bound_max_x:
            ax.set_xlim(left=bound_max_x)

        ax.set_ylim(top=min_y)
        if bound_max_x:
            ax.set_ylim(bottom=bound_max_y)

        norm = plt.Normalize(iterations.min(), iterations.max())
        sm = ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('viridis'))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.15)
        fig.colorbar(sm, cax=cax, label='Iterations')

    plt.savefig('figures/evals_overtime.pdf', dpi=450)


def plot_scatter_data(
        data,
        name,
        metric_x: Optional[str] = None,
        metric_y: Optional[str] = None,
        bound_max_x: Optional[float] = None,
        bound_max_y: Optional[float] = None,
):
    # metrics = experiments[0].optimization_config.objective.metrics
    #
    # metric_x = metrics[0] if metric_x is None else next(m for m in metrics if m.name == metric_x)
    # metric_y = metrics[1] if metric_y is None else next(m for m in metrics if m.name == metric_y)

    fig, axes = plt.subplots(len(experiments), figsize=(5, 5 * len(experiments)))

    for metrices in data:
        fig, ax = plt.subplots(1, figsize=(10, 5))
        #zip(axes, experiments):

        #data = experiment.fetch_data().df
        # values_x = data[data['metric_name'] == metric_x.name]['mean'].values
        # values_y = data[data['metric_name'] == metric_y.name]['mean'].values

        values_x = metrices[:,0] #if metric_x.lower_is_better else -1 * values_x
        values_y = metrices[:,1] #if metric_y.lower_is_better else -1 * values_y
        print(values_y)
        print(values_x)

        num_elements = values_x.size
        print(num_elements)
        iterations = np.arange(num_elements)

        ax.set_title(name)
        ax.scatter(values_x, values_y, c=iterations, alpha=0.8)

        ax.set_xlabel("val acc")
        ax.set_ylabel("num-params")

        # th_list = experiment.optimization_config.objective_thresholds
        # min_x = next(th.bound for th in th_list if th.metric.name == metric_x.name)
        # min_y = next(th.bound for th in th_list if th.metric.name == metric_y.name)
        #
        # ax.set_xlim(right=min_x)
        # if bound_max_x:
        #     ax.set_xlim(left=bound_max_x)
        #
        # ax.set_ylim(top=min_y)
        # if bound_max_x:
        #     ax.set_ylim(bottom=bound_max_y)

        norm = plt.Normalize(iterations.min(), iterations.max())
        sm = ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('viridis'))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.15)
        fig.colorbar(sm, cax=cax, label='Iterations')

    plt.show()#savefig('figures/evals_overtime.pdf', dpi=450)




def plot_aggregated_scatter_data(
        data,
        name,
        metric_x: Optional[str] = None,
        metric_y: Optional[str] = None,
        bound_max_x: Optional[float] = None,
        bound_max_y: Optional[float] = None,
):

    data=data[:5]
    fig, axes = plt.subplots(1, len(data), figsize=(5 * len(data), len(data)))
    setlabel=True
    for metrices, ax in zip(data,axes):


        #print("metrice",metrices)
        #fig, ax = plt.subplots(1, figsize=(10, 5))
            # zip(axes, experiments):

            # data = experiment.fetch_data().df
            # values_x = data[data['metric_name'] == metric_x.name]['mean'].values
            # values_y = data[data['metric_name'] == metric_y.name]['mean'].values

        values_x = metrices[:, 0]  # if metric_x.lower_is_better else -1 * values_x
        values_y = metrices[:, 1]  # if metric_y.lower_is_better else -1 * values_y
        print(values_y)
        print(values_x)

        num_elements = values_x.size
        print(num_elements)
        iterations = np.arange(num_elements)

        ax.set_title(name)
        ax.scatter(-values_x, values_y, c=iterations, alpha=0.8)
        ax.set_ylim(top=8)
        ax.set_ylim(bottom=2)
        ax.set_xlim(right=0)
        ax.set_xlim(left=110)
        if(setlabel == True):
            print(setlabel)
            ax.set_xlabel("val acc")
            ax.set_ylabel("num-params")
            setlabel=False


        # th_list = experiment.optimization_config.objective_thresholds
            # min_x = next(th.bound for th in th_list if th.metric.name == metric_x.name)
            # min_y = next(th.bound for th in th_list if th.metric.name == metric_y.name)
            #
            # ax.set_xlim(right=min_x)
            # if bound_max_x:
            #     ax.set_xlim(left=bound_max_x)
            #
            # ax.set_ylim(top=min_y)
            # if bound_max_x:
            #     ax.set_ylim(bottom=bound_max_y)

        norm = plt.Normalize(iterations.min(), iterations.max())
        sm = ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('viridis'))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.15)
        fig.colorbar(sm, cax=cax, label='Iterations')

    plt.show()#savefig(f'{path}/evals_overtime.pdf', dpi=450)

def plot_aggregated_scatter_adultdata(
        data,
        name,
        metric_x: Optional[str] = None,
        metric_y: Optional[str] = None,
        bound_max_x: Optional[float] = None,
        bound_max_y: Optional[float] = None,
):

    data=data[:5]
    fig, axes = plt.subplots(1, len(data), figsize=(5 * len(data), len(data)))
    setlabel=True
    for metrices, ax in zip(data,axes):


        #print("metrice",metrices)
        #fig, ax = plt.subplots(1, figsize=(10, 5))
            # zip(axes, experiments):

            # data = experiment.fetch_data().df
            # values_x = data[data['metric_name'] == metric_x.name]['mean'].values
            # values_y = data[data['metric_name'] == metric_y.name]['mean'].values

        values_x = metrices[:, 0]  # if metric_x.lower_is_better else -1 * values_x
        values_y = metrices[:, 1]  # if metric_y.lower_is_better else -1 * values_y
        print("values y\n", values_y)
        print("values x", values_x)

        num_elements = values_x.size
        print(num_elements)
        iterations = np.arange(num_elements)

        ax.set_title(name)
        ax.scatter(values_x, values_y, c=iterations, alpha=0.8)
        # ax.set_ylim(top=1)
        # ax.set_ylim(bottom=0)
        # ax.set_xlim(right=1)
        # ax.set_xlim(left=0)
        if(setlabel == True):
            print(setlabel)
            ax.set_xlabel("val error")
            ax.set_ylabel("dsp")
            setlabel=False


        # th_list = experiment.optimization_config.objective_thresholds
            # min_x = next(th.bound for th in th_list if th.metric.name == metric_x.name)
            # min_y = next(th.bound for th in th_list if th.metric.name == metric_y.name)
            #
            # ax.set_xlim(right=min_x)
            # if bound_max_x:
            #     ax.set_xlim(left=bound_max_x)
            #
            # ax.set_ylim(top=min_y)
            # if bound_max_x:
            #     ax.set_ylim(bottom=bound_max_y)

        norm = plt.Normalize(iterations.min(), iterations.max())
        sm = ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('viridis'))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.15)
        fig.colorbar(sm, cax=cax, label='Iterations')

    plt.show()

def plot_aggregated_scatter_nasdata(
        data,
        name,
        metric_x: Optional[str] = None,
        metric_y: Optional[str] = None,
        bound_max_x: Optional[float] = None,
        bound_max_y: Optional[float] = None,
):

    data=data[:5]
    fig, axes = plt.subplots(1, len(data), figsize=(5 * len(data), len(data)))
    setlabel=True
    for metrices, ax in zip(data,axes):

        values_x = metrices[:, 0]  # if metric_x.lower_is_better else -1 * values_x
        values_y = metrices[:, 1]  # if metric_y.lower_is_better else -1 * values_y
        print("values y\n", values_y)
        print("values x", values_x)

        num_elements = values_x.size
        print(num_elements)
        iterations = np.arange(num_elements)

        ax.set_title(name)
        ax.scatter(values_x, values_y, c=iterations, alpha=0.8)

        if(setlabel == True):
            print(setlabel)
            ax.set_xlabel("error")
            ax.set_ylabel("prediction time")
            setlabel=False

        norm = plt.Normalize(iterations.min(), iterations.max())
        sm = ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('viridis'))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.15)
        fig.colorbar(sm, cax=cax, label='Iterations')

    return plt
import os
path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\wiki_nsga_v2'
extension = 'txt'
os.chdir(path)
result = glob.glob('m*.{}'.format(extension))
#print(result)
experiments = [np.loadtxt(f) for f in result]
print(np.concatenate(experiments))
plot_aggregated_scatter_nasdata(experiments,'MODEHB_NSGA_V2')

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\wiki\\wiki_mobohb1'
extension = 'txt'
os.chdir(path)
result = glob.glob('m*.{}'.format(extension))
#print(result)
experiments = [np.loadtxt(f) for f in result]
print(np.concatenate(experiments))
plot_aggregated_scatter_nasdata(experiments,'MOBOHB')

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\wiki\\wiki_shemoa'
extension = 'txt'
os.chdir(path)
result = glob.glob('m*.{}'.format(extension))
#print(result)
experiments = [np.loadtxt(f) for f in result]
print(np.concatenate(experiments))
plot_aggregated_scatter_nasdata(experiments,'EASH')

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\wiki\\wiki_rs'
extension = 'txt'
os.chdir(path)
result = glob.glob('m*.{}'.format(extension))
#print(result)
experiments = [np.loadtxt(f) for f in result]
print(np.concatenate(experiments))
plot_aggregated_scatter_nasdata(experiments,'RS')

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\wiki_epsnet'
extension = 'txt'
os.chdir(path)
result = glob.glob('m*.{}'.format(extension))
#print(result)
experiments = [np.loadtxt(f) for f in result]
print(np.concatenate(experiments))
plot = plot_aggregated_scatter_nasdata(experiments,'WIKI_EPSNET_V2')



#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_mobohb\\flower_mobohb_experiments\\all_runs'
# extension = 'txt'
# os.chdir(path)
# result = glob.glob('*.{}'.format(extension))
# #print(result)
# experiments = [np.loadtxt(f) for f in result]
# print(np.concatenate(experiments))
# plot_aggregated_scatter_data(experiments,'MOBOHB')

# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\fashion_runs\\fashion_rs\\fashion_rs_experiments\\all_runs'
# extension = 'txt'
# os.chdir(path)
# result = glob.glob('*.{}'.format(extension))
# #print(result)
# experiments = [np.loadtxt(f) for f in result]
# print(np.concatenate(experiments))
# plot_aggregated_scatter_data(experiments,'RS')
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_rs\\metrics\\all_runs'
# extension = 'txt'
# os.chdir(path)
# result = glob.glob('*.{}'.format(extension))
# #print(result)
# experiments = [np.loadtxt(f) for f in result]
# print(np.concatenate(experiments))
# plot_aggregated_scatter_data(experiments,'RS')

# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\fashion_runs\\fashion_modehb_pc_sv2\\experiments\\all_runs'
# extension = 'txt'
# os.chdir(path)
# result = glob.glob('*.{}'.format(extension))
# #print(result)
# experiments = [np.loadtxt(f) for f in result]
# print(np.concatenate(experiments))
# plot_aggregated_scatter_data(experiments,'MODEHB')
# 
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_modehb\\pc_nds_sv2\\experiments\\metrices\\all_runs'
# extension = 'txt'
# os.chdir(path)
# result = glob.glob('*.{}'.format(extension))
# #print(result)
# experiments = [np.loadtxt(f) for f in result]
# print(np.concatenate(experiments))
# plot_aggregated_scatter_data(experiments,'MODEHB')

# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\adult_modehb\\all_runs'
# extension = 'txt'
# os.chdir(path)
# result = glob.glob('*.{}'.format(extension))
# #print(result)
# experiments = [np.loadtxt(f) for f in result]
# #print(experiments)
# plot_aggregated_scatter_adultdata(experiments,'MODEHB')
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\adult_rs\\all_runs'
# extension = 'txt'
# os.chdir(path)
# result = glob.glob('*.{}'.format(extension))
# #print(result)
# experiments = [np.loadtxt(f) for f in result]
# #print(experiments)
# plot_aggregated_scatter_adultdata(experiments,'RS')

