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
import sys
from loguru import logger
logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}


def _hypervolume_evolution_single(experiment: Experiment):
    assert isinstance(experiment.optimization_config, MultiObjectiveOptimizationConfig), \
        'experiment must have an optimization_config of type MultiObjectiveOptimizationConfig '

    th_list = experiment.optimization_config.objective_thresholds
    metrics = [th.metric for th in th_list]

    data = experiment.fetch_data().df
    data = [(-1 if m.lower_is_better else 1) * data[data['metric_name'] == m.name]['mean'].values for m in metrics]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(np.ascontiguousarray(np.asarray(data).T), device=device).float()
    logger.debug("objective ref point:{}",[(-1 if th.metric.lower_is_better else 1) * th.bound for th in th_list])
    hv = Hypervolume(torch.tensor([(-1 if th.metric.lower_is_better else 1) * th.bound for th in th_list], device=device))

    result = {
        'hypervolume': [0.0],
        'walltime': [0.0],
        'evaluations': [0.0]
    }

    logger.debug("data{}",data)

    for i in tqdm(range(1, data.shape[0]), desc=experiment.name):
        result['hypervolume'].append(hv.compute(data[:i][is_non_dominated(data[:i])]))
        logger.debug("hypervolume:{}",result['hypervolume'])
        try:
            result['walltime'].append(((experiment.trials[i - 1].time_completed - experiment.time_created).total_seconds()))
        except:
            result['walltime'].append(0)
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
    # fig, ax = plt.subplots()
    fig, axes = plt.subplots(4, figsize=(5, 5 * 4))

    # for exp in tqdm(experiments, desc="Hypervolume Evolution"):
    #     data = _hypervolume_evolution_single(exp)
    for data in datalist:
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
import  argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='', type=str, help='Timeout in sec. 0 -> no timeout')
args = parser.parse_args()
path = os.getcwd() + '/' + args.path
extension = 'pickle'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(result)

experiments = [load_experiment(path+'/'+f) for f in result]
# experiments = [
#         [
#             load_experiment(e) for e in glob.glob(f'{path}/{exp_name}_*_final.pickle')
#         ] for exp_name in experiments_names
#     ]
for exp in tqdm(experiments, desc="Hypervolume Evolution"):
    data = _hypervolume_evolution_single(exp)
    data.to_csv(path+'/'+exp.name+'.csv')


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

