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
        result['hypervolume'].append(hv.compute(data[:i][is_non_dominated(data[:i])]))
        result['walltime'].append(((experiment.trials[i - 1].time_completed - experiment.time_created).total_seconds()))
        result['evaluations'].append(i)

    return pd.DataFrame(result)
def plot_aggregated_scatter(
        path: str,
        experiment,
        metric_x: Optional[str] = None,
        metric_y: Optional[str] = None,
        bound_max_x: Optional[float] = None,
        bound_max_y: Optional[float] = None,
):


    metrics = experiments.optimization_config.objective.metrics

    metric_x = metrics[0] if metric_x is None else next(m for m in metrics if m.name == metric_x)
    metric_y = metrics[1] if metric_y is None else next(m for m in metrics if m.name == metric_y)

    #fig, axes = plt.subplots(1, len(experiments), figsize=(5 * len(experiments), len(experiments)))

    data = experiment.fetch_data().df
    values_x = data[data['metric_name'] == metric_x.name]['mean'].values
    values_y = data[data['metric_name'] == metric_y.name]['mean'].values

    values_x = values_x if metric_x.lower_is_better else -1 * values_x
    values_y = values_y if metric_y.lower_is_better else -1 * values_y


    return values_x,values_y


import  argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='results_msehvi', type=str, help='Timeout in sec. 0 -> no timeout')
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

