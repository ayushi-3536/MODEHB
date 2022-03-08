import glob
from typing import List, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from baselines import load_experiment
plt.style.use('ggplot')
import torch
from ax import Experiment, MultiObjectiveOptimizationConfig
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from matplotlib.cm import ScalarMappable
from tqdm import tqdm
from loguru import logger

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



    for i in tqdm(range(1, data.shape[0]), desc=experiment.name):
        result['hypervolume'].append(hv.compute(data[:i][is_non_dominated(data[:i])]))
        try:
            result['walltime'].append(((experiment.trials[i - 1].time_completed - experiment.time_created).total_seconds()))
        except:
            result['walltime'].append(0)
        result['evaluations'].append(i)

    return pd.DataFrame(result)

def plot_pareto_fronts(
        experiments: List[Experiment],
        ax,
        exp_type
):



    for exp in experiments:
        data = exp
        print(data)
        values_x = data[:,0]#[data['metric_name'] == metric_x.name]['mean'].values
        values_y = data[:,1]#['metric_name'] == metric_y.name]['mean'].values


        values = np.asarray([values_x, values_y]).T
        values = values[is_non_dominated(torch.as_tensor(values))]
        values = values[values[:, 0].argsort()]
        print(values)
        ax.plot(values[:, 0], values[:, 1], '-o', lw=3.0, label=exp_type)
        #plt.show()

    ax.set_xlabel('val-error')
    ax.set_ylabel('dsp')

    # ax.set_xlim(right=min_x)
    # if bound_max_x:
    #     ax.set_xlim(left=bound_max_x)
    #
    # ax.set_ylim(top=min_y)
    # if bound_max_x:
    #     ax.set_ylim(bottom=bound_max_y)

    ax.legend()
    plt.show()
    return ax


import os
import numpy as np
###############FLOWER ###########################################
fig, axes = plt.subplots(1, figsize=(10, 5))
all_exp = []
path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\adult\\adult_mobohb'
extension = 'txt'
os.chdir(path)
result = glob.glob('norm*.{}'.format(extension))
#print(result)
for f in result:
    exp = np.loadtxt(f)
    all_exp.extend(exp)

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\adult\\adult_modehb'
extension = 'txt'
os.chdir(path)
result = glob.glob('norm*.{}'.format(extension))
#print(result)
for f in result:
    exp = np.loadtxt(f)
    all_exp.extend(exp)
path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\adult\\adult_rs'
extension = 'txt'
os.chdir(path)
result = glob.glob('norm*.{}'.format(extension))
#print(result)
for f in result:
    exp = np.loadtxt(f)
    all_exp.extend(exp)
path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\adult\\adult_shemoa'
extension = 'txt'
os.chdir(path)
result = glob.glob('norm*.{}'.format(extension))
#print(result)
for f in result:
    exp = np.loadtxt(f)
    _hypervolume_evolution_single(exp)
    all_exp.extend(exp)

# data = all_exp
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# data = torch.tensor(np.ascontiguousarray(np.asarray(data)), device=device).float()
# ref_point = [-1.0, -1.0]
# hv = Hypervolume(torch.tensor(ref_point))
# pareto = data[is_non_dominated(-data)]
# hypervolume = hv.compute(pareto)
# print("pareto:{}",pareto)
# print("hypervolume:{}",hypervolume)
# with open("adult_metric.csv", "a+") as outfile:
#     outfile.write("hypervolume:{}\n".format(hypervolume))
#     outfile.write("ref_point:{}\n".format(ref_point))
#     outfile.write("pareto:{}\n".format(pareto))
#     outfile.close()


