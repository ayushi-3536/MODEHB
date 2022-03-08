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

from pygmo import hypervolume

data = np.array([[1.506455170178668546e+01, -4.5],
[5.250183072249459570e+01, -3.8],
[5.250183072249459570e+01, -6.7]])
rdata = data.copy()
print("data:{}",data)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = torch.tensor(np.ascontiguousarray(np.asarray(data)), device=device).float()
hv = Hypervolume(torch.tensor([-0.0, -8.0]))
for i in tqdm(range(1, data.shape[0]+1)):
    print(i)
    print(data[:i])
    pareto = data[:i][is_non_dominated(data[:i])]
    print("pareto:{}",pareto)
    print("hypervolume:{}",hv.compute(pareto))
data = np.array([[-1.506455170178668546e+01, 4.5],
[-5.250183072249459570e+01, 3.8],
[-5.250183072249459570e+01, 6.7]])
for i in tqdm(range(1, data.shape[0]+1)):
    print(i)
    print(data[:i])
    hv = hypervolume(data[:i])
    print(hv.contributions([0, 8]))
    # pareto = data[:i][is_non_dominated(data[:i])]
    # print("pareto:{}",pareto)
    # print("hypervolume:{}",hv.compute(pareto))


def _hypervolume_evolution_single(experiment: Experiment):
    assert isinstance(experiment.optimization_config, MultiObjectiveOptimizationConfig), \
        'experiment must have an optimization_config of type MultiObjectiveOptimizationConfig '

    th_list = experiment.optimization_config.objective_thresholds
    metrics = [th.metric for th in th_list]

    data = experiment.fetch_data().df
    data = [(-1 if m.lower_is_better else 1) * data[data['metric_name'] == m.name]['mean'].values for m in metrics]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(np.ascontiguousarray(np.asarray(data).T), device=device).float()

    hv = Hypervolume(torch.tensor([(-1 if th.metric.lower_is_better else 1) * th.bound for th in th_list], device=device))

    result = {
        'hypervolume': [0.0],
        'walltime': [0.0],
        'evaluations': [0.0]
    }



    for i in tqdm(range(1, data.shape[0]), desc=experiment.name):
        print(i)
        print(data[:i])
        result['hypervolume'].append(hv.compute(data[:i][is_non_dominated(data[:i])]))
        result['walltime'].append(((experiment.trials[i - 1].time_completed - experiment.time_created).total_seconds()))
        result['evaluations'].append(i)

    return pd.DataFrame(result)


def plot_hypervolume_evolution(
        #evolutions: Dict,
        experiments: List[Experiment],
):
    # fig, ax = plt.subplots()
    fig, axes = plt.subplots(4, figsize=(5, 5 * 4))

    for exp in tqdm(experiments, desc="Hypervolume Evolution"):
        data = _hypervolume_evolution_single(exp)

        #evolutions[exp.name] = data

        plot = axes[0].plot(data['walltime'], data['hypervolume'], '-', lw=3.0, label=exp.name)
        axes[0].set_xlabel('Walltime')
        axes[0].set_ylabel('Hypervolume')
        axes[0].legend()
        axes[0].set_xscale('log')

        if 'SHEMOA' in exp.name:
            axes[0].axvline(data['walltime'][50], color=plot[-1].get_color())
        if exp.name == 'EHVI':
            axes[0].axvline(data['walltime'][50], color=plot[-1].get_color())
        if exp.name == 'EASH':
            axes[0].axvline(data['walltime'][50], color=plot[-1].get_color())
        if exp.name == 'MOBOHB':
            axes[0].axvline(data['walltime'][50], color=plot[-1].get_color())
        if exp.name == 'BNC':
            axes[0].axvline(int(86400 * .35), color=plot[-1].get_color())
            axes[0].axvline(int(86400 * .75), color=plot[-1].get_color())


        axes[1].plot(data['evaluations'], data['hypervolume'], '-', lw=3.0, label=exp.name)
        axes[1].set_xlabel('Number of evaluations')
        axes[1].set_ylabel('Hypervolume')
        axes[1].legend()
        axes[1].set_xscale('log')

        if exp.name == 'MDEHVI':
            axes[1].axvline(50, color=plot[-1].get_color())
        if exp.name == 'BNC':
            axes[1].axvline(np.argmax(data['walltime'] > 86400 * .35), color=plot[-1].get_color())
            axes[1].axvline(np.argmax(data['walltime'] > 86400 * .75), color=plot[-1].get_color())




        ###############################
        plot = axes[2].plot(data['walltime'], data['hypervolume'], '-', lw=3.0, label=exp.name)
        axes[2].set_xlabel('Walltime')
        axes[2].set_ylabel('Hypervolume')
        axes[2].legend()

        if 'SHEMOA' in exp.name:
            axes[2].axvline(data['walltime'][50], color=plot[-1].get_color())
        if exp.name == 'EHVI':
            axes[2].axvline(data['walltime'][50], color=plot[-1].get_color())
        if exp.name == 'EASH':
            axes[2].axvline(data['walltime'][50], color=plot[-1].get_color())
        if exp.name == 'MOBOHB':
            axes[2].axvline(data['walltime'][50], color=plot[-1].get_color())
        if exp.name == 'BNC':
            axes[2].axvline(int(86400 * .35), color=plot[-1].get_color())
            axes[2].axvline(int(86400 * .75), color=plot[-1].get_color())


        axes[3].plot(data['evaluations'], data['hypervolume'], '-', lw=3.0, label=exp.name)
        axes[3].set_xlabel('Number of evaluations')
        axes[3].set_ylabel('Hypervolume')
        axes[3].legend()

        if exp.name == 'MDEHVI':
            axes[3].axvline(50, color=plot[-1].get_color())
        if 'BNC' in exp.name:
            axes[3].axvline(np.argmax(data['walltime'] > 86400 * .35), color=plot[-1].get_color())
            axes[3].axvline(np.argmax(data['walltime'] > 86400 * .75), color=plot[-1].get_color())



    plt.savefig('hypervolume.pdf', dpi=450)

def plot_hypervolume_fromdata(
        datalist
):

    name = 'SHEMOA'
    # fig, ax = plt.subplots()
    fig, axes = plt.subplots(4, figsize=(5, 5 * 4))

    # for exp in tqdm(experiments, desc="Hypervolume Evolution"):
    #     data = _hypervolume_evolution_single(exp)
    for data in datalist:
        #evolutions[exp.name] = data

        plot = axes[0].plot(data['walltime'], data['hypervolume'], '-', lw=3.0, label=name)
        axes[0].set_xlabel('Walltime')
        axes[0].set_ylabel('Hypervolume')
        axes[0].legend()
        axes[0].set_xscale('log')

        if 'SHEMOA' in name:
            axes[0].axvline(data['walltime'][50], color=plot[-1].get_color())
        if name == 'EHVI':
            axes[0].axvline(data['walltime'][50], color=plot[-1].get_color())
        if name == 'EASH':
            axes[0].axvline(data['walltime'][50], color=plot[-1].get_color())
        if name == 'MOBOHB':
            axes[0].axvline(data['walltime'][50], color=plot[-1].get_color())
        if name == 'BNC':
            axes[0].axvline(int(86400 * .35), color=plot[-1].get_color())
            axes[0].axvline(int(86400 * .75), color=plot[-1].get_color())


        axes[1].plot(data['evaluations'], data['hypervolume'], '-', lw=3.0, label=name)
        axes[1].set_xlabel('Number of evaluations')
        axes[1].set_ylabel('Hypervolume')
        axes[1].legend()
        axes[1].set_xscale('log')

        if name == 'MDEHVI':
            axes[1].axvline(50, color=plot[-1].get_color())
        if name == 'BNC':
            axes[1].axvline(np.argmax(data['walltime'] > 86400 * .35), color=plot[-1].get_color())
            axes[1].axvline(np.argmax(data['walltime'] > 86400 * .75), color=plot[-1].get_color())




        ###############################
        plot = axes[2].plot(data['walltime'], data['hypervolume'], '-', lw=3.0, label=name)
        axes[2].set_xlabel('Walltime')
        axes[2].set_ylabel('Hypervolume')
        axes[2].legend()

        if 'SHEMOA' in name:
            axes[2].axvline(data['walltime'][50], color=plot[-1].get_color())
        if name == 'EHVI':
            axes[2].axvline(data['walltime'][50], color=plot[-1].get_color())
        if name == 'EASH':
            axes[2].axvline(data['walltime'][50], color=plot[-1].get_color())
        if name == 'MOBOHB':
            axes[2].axvline(data['walltime'][50], color=plot[-1].get_color())
        if name == 'BNC':
            axes[2].axvline(int(86400 * .35), color=plot[-1].get_color())
            axes[2].axvline(int(86400 * .75), color=plot[-1].get_color())


        axes[3].plot(data['evaluations'], data['hypervolume'], '-', lw=3.0, label=name)
        axes[3].set_xlabel('Number of evaluations')
        axes[3].set_ylabel('Hypervolume')
        axes[3].legend()

        if name == 'MDEHVI':
            axes[3].axvline(50, color=plot[-1].get_color())
        if 'BNC' in name:
            axes[3].axvline(np.argmax(data['walltime'] > 86400 * .35), color=plot[-1].get_color())
            axes[3].axvline(np.argmax(data['walltime'] > 86400 * .75), color=plot[-1].get_color())



    plt.savefig('hypervolume.pdf', dpi=450)


def plot_aggregated_hypervolume_time(experiments: Dict, path: str, log=False, valid_experiments=[]):

    experiments_by_type = defaultdict(list)
    for exp, val in experiments.items():
        experiments_by_type[exp.split('_')[0]].append(deepcopy(val))

    fig, axes = plt.subplots(1, figsize=(10, 5))

    for exp_type, values in experiments_by_type.items():

        if exp_type not in valid_experiments:
            continue

        if exp_type != 'BNC':
            timechange = np.mean([values[i]['walltime'][{
                'MDEHVI': 10,
                'MOBOHB': 10,
                'EASH': 25,
                'RS': 0,
                'MOSHBANANAS': 20,
            }[exp_type]] for i in range(len(values))])/3600
        else:
            #timechange = 23 * 3600 * (.35)
            timechange = 23  * (.35)
        df = None
        for i in range(len(values)):
            values[i]['walltime'] = pd.to_timedelta(values[i]['walltime'], unit='s')
            values[i] = values[i].resample('1h', on='walltime').max()[['hypervolume']]
            df = values[i] if df is None else pd.merge(df, values[i], how='outer', on='walltime')

        df.columns = [f'hypervolume_{i}' for i in range(len(values))]
        df = df.pad()

        walltime = df.index.astype('timedelta64[h]').values
        mean = df.mean(axis=1).values
        std = df.std(axis=1).values
        axes.set_xlim(0, 25)
        plot = axes.plot(walltime, mean, '-', lw=3.0, label=exp_type)
        plt.axvline(timechange, color=plot[-1].get_color())
        #axes.fill_between(walltime, mean + 2 * std, mean - 2 * std, alpha=0.5)

    axes.set_xlabel('Walltime')
    axes.set_ylabel('Hypervolume')
    axes.legend()
    if log:
        axes.set_xscale('log')

    #plt.savefig(f'{path}/aggregated_time_log_{str(log).lower()}.pdf', dpi=450)
    plt.savefig(f'{path}/aggregated_time_log_{str(log).lower()}.jpg', dpi=450)


def plot_aggregated_hypervolume_evaluations_data(data, path: str,axes,exp_type, log=False, valid_experiments=[]):

    # experiments_by_type = defaultdict(list)
    # for exp, val in experiments.items():
    #     experiments_by_type[exp.split('_')[0]].append(val)


    #for exp_type, values in experiments_by_type.items():

        values = data

        df = None
        for i in range(len(values)):
            values[i] = values[i].set_index('evaluations')[['hypervolume']]
            df = values[i] if df is None else pd.merge(df, values[i], how='outer', on='evaluations')

        df.columns = [f'hypervolume_{i}' for i in range(len(values))]
        df = df.pad()

        #print(exp_type)
        #print(df.values[:4])
        #print(np.std(df.values[:4], axis=1))

        evaluations = df.index.values
        mean = df.mean(axis=1).values
        std = df.std(axis=1).values
        plot = axes.plot(evaluations, mean, '-', lw=3.0, label=exp_type)
        plt.axvline(evaluations[{
            'MDEHVI': 10,
            'BNC': 0,
            'MOBOHB': 10,
            'MODEHB': 10,
            'EASH': 25,
            'RS': 0,
            'MOSHBANANAS': 20,
        }[exp_type]], color=plot[-1].get_color())
        #axes.fill_between(evaluations, mean + 2 * std, mean - 2 * std, alpha=0.5)

        axes.set_xlabel('Evaluations')
        axes.set_ylabel('Hypervolume')
        axes.legend()
        if log:
            axes.set_xscale('log')
        return axes
    #plt.savefig(f'{path}/aggregated_evaluations_log_{str(log).lower()}.pdf', dpi=450)
    #plt.savefig(f'{path}/aggregated_evaluations_log_{str(log).lower()}.jpg', dpi=450)

def plot_aggregated_hypervolume_time_data(data, path: str,axes,exp_type, log=False, valid_experiments=[]):

    # experiments_by_type = defaultdict(list)
    # for exp, val in experiments.items():
    #     experiments_by_type[exp.split('_')[0]].append(deepcopy(val))



                values=data

                # if exp_type not in valid_experiments:
                #     continue
                # #print("#############################VALUES###################################")
                # #print(values)

                ##print(values[0]['walltime'])
                if exp_type != 'BNC':
                    timechange = np.mean([values[i]['walltime'][{
                        'MDEHVI': 10,
                        'MOBOHB': 10,
                        'MODEHB': 10,
                        'EASH': 25,
                        'RS': 25,
                        'MOSHBANANAS': 20,
                    }[exp_type]] for i in range(len(values))])/3600
                else:
                    #timechange = 23 * 3600 * (.35)
                    timechange = 23  * (.35)
                df = None
                #print("timechange",timechange)
                for i in range(len(values)):
                    #print("##########################BEFORE###########################")
                    #print(values[i]['walltime'])
                    values[i]['walltime'] = pd.to_timedelta(values[i]['walltime'], unit='s')
                    #print(values[i]['walltime'])
                    values[i] = values[i].resample('1h', on='walltime').max()[['hypervolume']]
                    #print("values[i] after resampling")
                    #print(values[i])
                    df = values[i] if df is None else pd.merge(df, values[i], how='outer', on='walltime')
                    #print("df after merge")
                    #print(df)

                df.columns = [f'hypervolume_{i}' for i in range(len(values))]
                #print(df.columns)

                df = df.pad()

                walltime = df.index.astype('timedelta64[h]').values
                #print("walltime shape",walltime.shape)
                mean = df.mean(axis=1).values
                #print("mean",mean)
                std = df.std(axis=1).values
                #print("std",std)
                axes.set_xlim(0, 25)
                plot = axes.plot(walltime, mean, '-', lw=3.0, label=exp_type)
                plt.axvline(timechange, color=plot[-1].get_color())
                axes.fill_between(walltime, mean + 0.5*std, mean - 0.5*std, alpha=0.5)

                axes.set_xlabel('Walltime')
                axes.set_ylabel('Hypervolume')
                axes.legend()
                if log:
                    axes.set_xscale('log')

                #plt.savefig(f'{path}/aggregated_time_log_{str(log).lower()}.pdf', dpi=450)
                return axes#plt.savefig(f'{path}/aggregated_time_log_{str(log).lower()}.jpg', dpi=450)

import os

###############NAS BENCH FLOWER ###########################################
fig, axes = plt.subplots(1, figsize=(10, 5))

path = 'C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\flower_shemoa\\shemoa_flower'
extension = 'csv'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
#print(result)
experiments = [pd.read_csv(f) for f in result]
axes = plot_aggregated_hypervolume_time_data(experiments,path,axes,'EASH')
# #axes = plot_aggregated_hypervolume_evaluations_data(experiments,path,axes,'MDEHVI')
# path = 'C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\' + 'mosh_csv'
# extension = 'csv'
# os.chdir(path)
# result = glob.glob('*.{}'.format(extension))
# #print(result)
# experiments = [pd.read_csv(f) for f in result]
# axes = plot_aggregated_hypervolume_time_data(experiments,path,axes,'MOSHBANANAS')
path = 'C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\mobohb_flower'
extension = 'csv'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
#print(result)
experiments = [pd.read_csv(f) for f in result]
axes = plot_aggregated_hypervolume_time_data(experiments,path,axes,'MOBOHB')
path = 'C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\csv'
extension = 'csv'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
experiments = [pd.read_csv(f) for f in result]
for e in experiments:
    #print(e['walltime'])
    e['walltime']= np.cumsum(np.array(e['walltime']), dtype=np.float64)
    #print("after cumsum")
    #print(e['walltime'])
    ##print("cumsum",))

axes = plot_aggregated_hypervolume_time_data(experiments,path,axes,'MODEHB')


path = 'C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\flower_rs\\rs_flower'
extension = 'csv'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
experiments = [pd.read_csv(f) for f in result]

axes = plot_aggregated_hypervolume_time_data(experiments,path,axes,'RS')
# #axes = plot_aggregated_hypervolume_evaluations_data(experiments,path,axes,'MODEHB')

plt.savefig(f'aggregated_time.pdf', dpi=450)

###############NAS BENCH FASHION ##########################################
# fig, axes = plt.subplots(1, figsize=(10, 5))
# axes = plot_aggregated_hypervolume_time_data(experiments,path,axes,'EASH')
# #axes = plot_aggregated_hypervolume_evaluations_data(experiments,path,axes,'EASH')
#
# path = 'C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\' + 'results_msehvi'
# extension = 'csv'
# os.chdir(path)
# result = glob.glob('*.{}'.format(extension))
# #print(result)
# experiments = [pd.read_csv(f) for f in result]
# axes = plot_aggregated_hypervolume_time_data(experiments,path,axes,'MDEHVI')
# #axes = plot_aggregated_hypervolume_evaluations_data(experiments,path,axes,'MDEHVI')
# path = 'C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\' + 'mosh_csv'
# extension = 'csv'
# os.chdir(path)
# result = glob.glob('*.{}'.format(extension))
# #print(result)
# experiments = [pd.read_csv(f) for f in result]
# axes = plot_aggregated_hypervolume_time_data(experiments,path,axes,'MOSHBANANAS')
# path = 'C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\' + 'mobohb_csv'
# extension = 'csv'
# os.chdir(path)
# result = glob.glob('*.{}'.format(extension))
# #print(result)
# experiments = [pd.read_csv(f) for f in result]
# axes = plot_aggregated_hypervolume_time_data(experiments,path,axes,'MOBOHB')
# path = 'C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\' + 'rs'
# extension = 'csv'
# os.chdir(path)
# result = glob.glob('*.{}'.format(extension))
# #print(result)
# experiments = [pd.read_csv(f) for f in result]
# axes = plot_aggregated_hypervolume_time_data(experiments,path,axes,'RS')
#
#
# path = 'C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\res_modehb\\dehb_run\\final'
# extension = 'csv'
# os.chdir(path)
# result = glob.glob('*.{}'.format(extension))
# #print(result)
# experiments = [pd.read_csv(f) for f in result]
# axes = plot_aggregated_hypervolume_time_data(experiments,path,axes,'MODEHB')
# #axes = plot_aggregated_hypervolume_evaluations_data(experiments,path,axes,'MODEHB')
#
# plt.savefig(f'aggregated_time_mean.pdf', dpi=450)

def plot_aggregated_hypervolume_evaluations(data, path: str, log=False, valid_experiments=[]):

    # experiments_by_type = defaultdict(list)
    # for exp, val in experiments.items():
    #     experiments_by_type[exp.split('_')[0]].append(val)

    fig, axes = plt.subplots(1, figsize=(10, 5))

    #for exp_type, values in experiments_by_type.items():
    exp_type = 'SHEMOA'
    for values in data:

        if exp_type not in valid_experiments:
            continue

        df = None
        for i in range(len(values)):
            values[i] = values[i].set_index('evaluations')[['hypervolume']]
            df = values[i] if df is None else pd.merge(df, values[i], how='outer', on='evaluations')

        df.columns = [f'hypervolume_{i}' for i in range(len(values))]
        df = df.pad()

        #print(exp_type)
        #print(df.values[:4])
        #print(np.std(df.values[:4], axis=1))

        evaluations = df.index.values
        mean = df.mean(axis=1).values
        std = df.std(axis=1).values
        plot = axes.plot(evaluations, mean, '-', lw=3.0, label=exp_type)
        plt.axvline(evaluations[{
            'MDEHVI': 10,
            'BNC': 0,
            'MOBOHB': 10,
            'EASH': 25,
            'RS': 0,
            'MOSHBANANAS': 20,
        }[exp_type]], color=plot[-1].get_color())
        #axes.fill_between(evaluations, mean + 2 * std, mean - 2 * std, alpha=0.5)

    axes.set_xlabel('Evaluations')
    axes.set_ylabel('Hypervolume')
    axes.legend()
    if log:
        axes.set_xscale('log')

    #plt.savefig(f'{path}/aggregated_evaluations_log_{str(log).lower()}.pdf', dpi=450)
    plt.savefig(f'{path}/aggregated_evaluations_log_{str(log).lower()}.jpg', dpi=450)



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
