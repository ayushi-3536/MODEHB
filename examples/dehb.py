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

def _hypervolume_evolution_single_dehb(data,cum_time):
    #assert isinstance(experiment.optimization_config, MultiObjectiveOptimizationConfig), \
        #'experiment must have an optimization_config of type MultiObjectiveOptimizationConfig '



    # data = experiment.fetch_data().df
    #
    # data = [data[data['metric_name'] == m.name]['mean'].values for m in metrics]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data = torch.tensor(np.ascontiguousarray(np.asarray(data).T), device=device).float()

    hv = Hypervolume(torch.tensor([-0.0,-8.0], device=device))

    result = {
        'hypervolume': [0.0],
        'walltime': [0.0],
        'evaluations': [0.0]
    }

    print("data shape,",data.shape)
    print("time shape",cum_time.shape)
    t=0
    for i in tqdm(range(1, data.shape[0]), desc='data extraction'):
        print(i)
        print("nds data:{}",data[:i][is_non_dominated(data[:i])])
        print("hypervol:{}",hv.compute(data[:i][is_non_dominated(data[:i])]))
        result['hypervolume'].append(hv.compute(data[:i][is_non_dominated(data[:i])]))
        t=t + cum_time[i]
        result['walltime'].append(t)
        result['evaluations'].append(i)
    #result['walltime'] = cum_time
    return pd.DataFrame(result)
#
# path = 'C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\msehvi_final\\results\\extra'
# extension = 'pickle'
# os.chdir(path)
# result = glob.glob('*.{}'.format(extension))
# print(result)
# #reg_hv,times,f_pf_modehb = graph(path,result)
# experiments = [load_experiment(path+'/'+f) for f in result]
# # experiments = [
# #         [
# #             load_experiment(e) for e in glob.glob(f'{path}/{exp_name}_*_final.pickle')
# #         ] for exp_name in experiments_names
# #     ]
# for exp in tqdm(experiments, desc="Hypervolume Evolution"):
#     data = _hypervolume_evolution_single(exp)
#     data.to_csv(path+'/'+exp.name+'.csv')

import os

path = 'C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\flower_dehb'
extension = 'txt'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(result)
#reg_hv,times,f_pf_modehb = graph(path,result)
#r = [np.loadtxt(path + '\\' + f) for f in result]
for idx,f in enumerate(result):
        r = np.loadtxt(path + '\\' + f)
        r = np.array(r)
        #print(r)
        print("r:{}",r.shape)
        data = np.array(-r[:,1])
        data = np.vstack((data, np.array(-r[:,0])))
        #print(data)
        print(data.shape)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data = torch.tensor(np.ascontiguousarray(np.asarray(data).T), device=device).float()
        print("data:{}",data)
        time = np.array(r[:,2])
        print("time:{}",time.shape)
        #print("data:{}",data)
        #ax.legend(loc="lower right")
        data = _hypervolume_evolution_single_dehb(data,time)
        data.to_csv(path+'\\full_'+str(idx)+'.csv')
