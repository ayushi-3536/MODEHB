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
import  argparse
import os
from loguru import logger
import sys
logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}


def get_metrices(
        experiment,
        path,
        idx,
        metric_x: Optional[str] = None,
        metric_y: Optional[str] = None,
        bound_max_x: Optional[float] = None,
        bound_max_y: Optional[float] = None,
):

    metrics = experiment.optimization_config.objective.metrics
    logger.debug("metrics:{}",metrics)
    metric_x = metrics[0] if metric_x is None else next(m for m in metrics if m.name == metric_x)
    metric_y = metrics[1] if metric_y is None else next(m for m in metrics if m.name == metric_y)
    logger.debug("metric_x:{}", metric_x)
    logger.debug("metric_y:{}", metric_y)

    th_list = experiment.optimization_config.objective_thresholds
    logger.debug("th_list:{}",th_list)
    logger.debug("th with sign:{}",[(-1 if th.metric.lower_is_better else 1) * th.bound for th in th_list])
    hv = Hypervolume(torch.tensor([(-1 if th.metric.lower_is_better else 1) * th.bound for th in th_list], device=torch.device("cuda:0")))


    data = exp.fetch_data().df
    values_x = data[data['metric_name'] == metric_x.name]['mean'].values
    values_y = data[data['metric_name'] == metric_y.name]['mean'].values

    maximizator = [-1 if metric_x.lower_is_better else 1, -1 if metric_y.lower_is_better else 1]

    values = np.asarray([values_x, values_y]).T
    with open(os.path.join(path, "metric_{}.txt".format(idx)), 'w') as f:
        np.savetxt(f, values)

    values = values[is_non_dominated(torch.as_tensor(values * maximizator))]
    values = values[values[:, 0].argsort()]

    with open(os.path.join(path, "sortmetric_{}.txt".format(idx)), 'w') as f:
        np.savetxt(f, values)



parser = argparse.ArgumentParser()
parser.add_argument('--path', default='', type=str, help='Timeout in sec. 0 -> no timeout')
args = parser.parse_args()
path = os.getcwd() + '/' + args.path
extension = 'pickle'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
logger.debug("path:{},result:{}",path,result)

experiments = [load_experiment(path+'/'+f) for f in result]
# experiments = [
#         [
#             load_experiment(e) for e in glob.glob(f'{path}/{exp_name}_*_final.pickle')
#         ] for exp_name in experiments_names
#     ]
for idx,exp in enumerate(tqdm(experiments, desc="Metrices generation")):
    get_metrices(exp,path,idx)


