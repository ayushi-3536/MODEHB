import glob
from copy import deepcopy
from typing import List, Optional, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from baselines import load_experiment
from .normalization_transform import GaussianTransform
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
    data = [data[data['metric_name'] == m.name]['mean'].values for m in metrics]
    data = np.ascontiguousarray(np.asarray(data).T)
    psi = GaussianTransform(data)
    data = psi.transform(data)
    print(data)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(-data, device=device).float()

    print(data)

    logger.debug("objective ref point:{}",[(-1 if th.metric.lower_is_better else 1) * th.bound for th in th_list])
    #hv = Hypervolume(torch.tensor([(-1 if th.metric.lower_is_better else 1) * th.bound for th in th_list], device=device))
    hv = Hypervolume(torch.tensor([-1.0,-1.0], device=device))

    result = {
        'hypervolume': [0.0],
        'walltime': [0.0],
        'evaluations': [0.0]
    }

    print(is_non_dominated(data))
    pareto = data[is_non_dominated(data)]
    print(pareto)

    print("hv", hv.compute(pareto))


    for i in tqdm(range(1, data.shape[0]), desc=experiment.name):
        result['hypervolume'].append(hv.compute(data[:i][is_non_dominated(data[:i])]))
        try:
            result['walltime'].append(((experiment.trials[i - 1].time_completed - experiment.time_created).total_seconds()))
        except:
            result['walltime'].append(0)
        result['evaluations'].append(i)

    return pd.DataFrame(result)



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
    data.to_csv(path+'/norm_'+exp.name+'.csv')



