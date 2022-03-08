import glob
from copy import deepcopy
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('ggplot')
import torch
from ax import Experiment, MultiObjectiveOptimizationConfig
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from collections import defaultdict
from tqdm import tqdm


def get_experiment_list(path_list):
    all_exp = []
    for path in path_list:
        extension = 'txt'
        os.chdir(path)
        result = glob.glob('metric*.{}'.format(extension))
        for f in result:
            all_exp.extend(np.loadtxt(f))
    return all_exp

def get_max_hv(path_list,ref_point=None):
    data = get_experiment_list(path_list)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(np.ascontiguousarray(np.asarray(data)), device=device).float()
    if ref_point == None:
        ref_point = [-1.0, -1.0]
    print(ref_point)
    hv = Hypervolume(torch.tensor(ref_point))
    data = -data
    pareto = data[is_non_dominated(data)]
    hypervolume = hv.compute(pareto)
    print("pareto:{}",pareto)
    print("hypervolume:{}",hypervolume)

    return hypervolume

def _get_experiments(path, extension):
    extension = 'csv'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    print(result)
    experiments = [pd.read_csv(f) for f in result]
    return experiments


def plot_aggregated_norm_hypervolume(data,
                                     axes,
                                     exp_type,
                                     maxhv,
                                     x_limit = 3600,
                                     log=False, ):

    values = data
    timechange = np.mean([values[i]['walltime'][{
            'MDEHVI': 0,
            'MOBOHB': 0,

            'MODEHB_NDS_V2': 0,
            'MODEHB_NSGA_V2': 0,
            'MODEHB_NSGA_V3': 0,
            'MODEHB_NSGA_V4': 0,
            'MODEHB_EPSNET_V2': 0,
            'MODEHB_EPSNET_V3': 0,
            'MODEHB_EPSNET_V4': 0,
            'EASH': 0,
            'RS': 0,
            'MOSHBANANAS': 0,
    }[exp_type]] for i in range(len(values))])

    df = None
    for i in range(len(values)):
        values[i]['walltime'] = pd.to_timedelta(values[i]['walltime'], unit='s')
        values[i] = values[i].resample('s', on='walltime').max()[['hypervolume']]
        df = values[i] if df is None else pd.merge(df, values[i], how='outer', on='walltime')

    df.columns = [f'hypervolume_{i}' for i in range(len(values))]
    df = df.pad()

    mean = df.mean(axis=1).values
    std = df.std(axis=1).values
    axes.set_xlim(0, x_limit)
    logmean = np.log10(maxhv - mean)
    print("maxhv:",maxhv,"mean:",mean,"logmean",logmean)
    mean=logmean
    plot = axes.plot(mean, '-', lw=3.0, label=exp_type)
    plt.axvline(timechange, color=plot[-1].get_color())
    axes.fill_between(mean + 0.5 * std, mean - 0.5 * std, alpha=0.5)

    axes.set_xlabel('Walltime')
    axes.set_ylabel('Log10 Hypervolume Difference')
    # axes.set_xscale('log')
    # plt.xlim([1, 10 ** 4])

    axes.legend()
    if log:
        axes.set_xscale('log')
    return axes


def plot_aggregated_hypervolume_time_data(data, axes, exp_type, log=False):


    values = data
    if exp_type != 'BNC':
        timechange = np.mean([values[i]['walltime'][{
            'MSEHVI': 0,
            'MOBOHB': 0,
            'MODEHB': 0,
            'MODEHB_NDS_V2': 0,
            'MODEHB_EPSNET_V2': 0,
            'MODEHB_MAXHV_V2': 0,
            'MODEHB_NDSV3': 0,
            'MODEHB_HVV3': 0,
            'EASH': 0,
            'RS': 0,
            'MOSHBANANAS': 0,
        }[exp_type]] for i in range(len(values))])
    else:
        # timechange = 23 * 3600 * (.35)
        timechange = 23 * (.35)
    df = None
    # print("timechange",timechange)
    for i in range(len(values)):
        values[i]['walltime'] = pd.to_timedelta(values[i]['walltime'], unit='s')
        values[i] = values[i].resample('1h', on='walltime').max()[['hypervolume']]
        df = values[i] if df is None else pd.merge(df, values[i], how='outer', on='walltime')

    df.columns = [f'hypervolume_{i}' for i in range(len(values))]
    df = df.pad()
    walltime = df.index.astype('timedelta64[h]').values
    mean = df.mean(axis=1).values
    std = df.std(axis=1).values
    axes.set_xlim(0, 24)
    plot = axes.plot(walltime, mean, '-', lw=3.0, label=exp_type)
    plt.axvline(timechange, color=plot[-1].get_color())
    axes.fill_between(walltime, mean + 0.5 * std, mean - 0.5 * std, alpha=0.5)

    axes.set_xlabel('Walltime')
    axes.set_ylabel('Hypervolume')
    axes.legend()
    if log:
        axes.set_xscale('log')
    return axes

def plot_aggregated_hypervolume_adult(data, axes, exp_type, log=False):


    values = data
    if exp_type != 'BNC':
        timechange = np.mean([values[i]['walltime'][{
            'MDEHVI': 0,
            'MOBOHB': 0,
            'MODEHB': 0,
            'MODEHB_NDSV2': 0,
            'MODEHB_NDSV3': 0,
            'MODEHB_HVV3': 0,
            'MODEHB_EPSNET': 0,
            'EASH': 0,
            'RS': 0,
            'MOSHBANANAS': 0,
        }[exp_type]] for i in range(len(values))]) / 3600
    else:
        # timechange = 23 * 3600 * (.35)
        timechange = 23 * (.35)
    df = None
    # print("timechange",timechange)
    for i in range(len(values)):
        values[i]['walltime'] = pd.to_timedelta(values[i]['walltime'], unit='s')
        values[i] = values[i].resample('s', on='walltime').max()[['hypervolume']]
        df = values[i] if df is None else pd.merge(df, values[i], how='outer', on='walltime')

    df.columns = [f'hypervolume_{i}' for i in range(len(values))]

    df = df.pad()

    walltime = df.index.astype('timedelta64[h]').values
    mean = df.mean(axis=1).values
    std = df.std(axis=1).values
    axes.set_xlim(0, 3600)
    plot = axes.plot( mean, '-', lw=3.0, label=exp_type)
    plt.axvline(timechange, color=plot[-1].get_color())
    axes.fill_between(walltime, mean + 0.5 * std, mean - 0.5 * std, alpha=0.5)

    axes.set_xlabel('Walltime')
    axes.set_ylabel('Hypervolume')
    axes.legend()
    if log:
        axes.set_xscale('log')
    return axes

def plot_aggregated_hypervolume_nas(data,  axes, exp_type,maxhv, log=False):

    values = data
    print(len(values))

    ##print(values[0]['walltime'])
    if exp_type != 'BNC':
        timechange = np.mean([values[i]['walltime'][{
            'MDEHVI': 0,
            'MOBOHB': 0,
            'MODEHB': 0,
            'MODEHB_NSGA_V2': 0,
            'MODEHB_NSGA_V3': 0,
            'MODEHB_NSGA_V4': 0,
            'MODEHB_EPSNET_V2':0,
            'MODEHB_EPSNET_V3': 0,
            'MODEHB_EPSNET_V4': 0,
            'EASH': 0,
            'RS': 0,
            'MOSHBANANAS': 0,
        }[exp_type]] for i in range(len(values))])
    else:
        # timechange = 23 * 3600 * (.35)
        timechange = 23 * (.35)

    df = None
    # print("timechange",timechange)
    for i in range(len(values)):
        values[i]['walltime'] = pd.to_timedelta(values[i]['walltime'], unit='s')
        values[i] = values[i].resample('1h', on='walltime').max()[['hypervolume']]
        df = values[i] if df is None else pd.merge(df, values[i], how='outer', on='walltime')

    df.columns = [f'hypervolume_{i}' for i in range(len(values))]
    df = df.pad()
    walltime = df.index.astype('timedelta64[h]').values
    mean = df.mean(axis=1).values
    std = df.std(axis=1).values
    mean = np.log10(maxhv-mean)
    axes.set_xlim(0, 240)
    plot = axes.plot(walltime, mean, '-', lw=3.0, label=exp_type)
    plt.axvline(timechange, color=plot[-1].get_color())
    #axes.fill_between(walltime, mean + 0.5 * std, mean - 0.5 * std, alpha=0.5)

    axes.set_xlabel('Walltime')
    axes.set_ylabel('Log Hypervolume Diff')
    # axes.set_yscale('log')
    axes.legend()
    # axes.set_xscale('log')
    df.columns = [f'hypervolume_{i}' for i in range(len(values))]
    # print(df.columns)

    # plt.savefig(f'{path}/aggregated_time_log_{str(log).lower()}.pdf', dpi=450)
    return axes  # plt.savefig(f'{path}/aggregated_time_log_{str(log).lower()}.jpg', dpi=450)


import os

###############FLOWER ###########################################
fig, axes = plt.subplots(1, figsize=(10, 5))
FILE_EXTENSION = 'csv'
path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\flower_shemoa'
axes = plot_aggregated_hypervolume_time_data(_get_experiments(path, FILE_EXTENSION), axes, 'EASH')

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_mobohb\\hv'
axes = plot_aggregated_hypervolume_time_data(_get_experiments(path, FILE_EXTENSION), axes, 'MOBOHB')

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_modehb\\pc_nds_sv2\\hv'
axes = plot_aggregated_hypervolume_time_data(_get_experiments(path, FILE_EXTENSION), axes, 'MODEHB_NDS_V2')


path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_mosh\\hv'
axes = plot_aggregated_hypervolume_time_data(_get_experiments(path, FILE_EXTENSION), axes, 'MOSHBANANAS')

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_msehvi\\hv'
axes = plot_aggregated_hypervolume_time_data(_get_experiments(path, FILE_EXTENSION), axes, 'MSEHVI')


path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\flower_epsnet'
axes = plot_aggregated_hypervolume_time_data(_get_experiments(path, FILE_EXTENSION), axes, 'MODEHB_EPSNET_V2')

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_rs\\hv'
axes = plot_aggregated_hypervolume_time_data(_get_experiments(path, FILE_EXTENSION), axes, 'RS')

plt.savefig(f'aggregated_time.pdf', dpi=450)

############################################ FASHION  ###########################################
# fig, axes = plt.subplots(1, figsize=(10, 5))
# FILE_EXTENSION = 'csv'
# # path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\fashion_runs\\fashion_shemoa\\hv'
# # axes = plot_aggregated_hypervolume_time_data(_get_experiments(path, FILE_EXTENSION), axes, 'EASH')
# #
# # path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\fashion_runs\\fashion_mobohb\\hv'
# # axes = plot_aggregated_hypervolume_time_data(_get_experiments(path, FILE_EXTENSION), axes, 'MOBOHB')
#
#
#
# # path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\fashion_runs\\fashion_moshbananas\\hv'
# # axes = plot_aggregated_hypervolume_time_data(_get_experiments(path, FILE_EXTENSION), axes, 'MOSHBANANAS')
# #
# # path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\fashion_runs\\fashion_msehvi\\hv'
# # axes = plot_aggregated_hypervolume_time_data(_get_experiments(path, FILE_EXTENSION), axes, 'MSEHVI')
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\fashion_runs\\fashion_modehb_pc_sv2\\hv'
# axes = plot_aggregated_hypervolume_time_data(_get_experiments(path, FILE_EXTENSION), axes, 'MODEHB_MAXHV_V2')
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\fashion_nsga_v2'
# axes = plot_aggregated_hypervolume_time_data(_get_experiments(path, FILE_EXTENSION), axes, 'MODEHB_NDS_V2')
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\fashion_epsnet_v2'
# axes = plot_aggregated_hypervolume_time_data(_get_experiments(path, FILE_EXTENSION), axes, 'MODEHB_EPSNET_V2')
# #
# # path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\fashion_runs\\fashion_rs\\hv'
# # axes = plot_aggregated_hypervolume_time_data(_get_experiments(path, FILE_EXTENSION), axes, 'RS')
#
# plt.savefig(f'aggregated_time.pdf', dpi=450)
#


############# NAS BENCH 201 ##########################################
# fig, axes = plt.subplots(1, figsize=(10, 5))
# FILE_EXTENSION = 'csv'
# # X_LIMIT=3600
#
# rs_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\nas\\rs_nas'
# modehb_ndsv2_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\nas\\modehb_nas_ndsv2'
# mobohb_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\nas\\mobohb_nas1'
# shemoa_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\nas\\shemoa_nas1'
# modehb_epsnetv2_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\nas_epsnet'
# modehb_epsnetv3_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\nas_epsnet_v3'
# modehb_nsgav2_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\nas_nsga_v2'
# modehb_nsgav3_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\nas_nsga_v3'
#
# path=[
#     rs_path, mobohb_path, shemoa_path,modehb_ndsv2_path,modehb_nsgav3_path,
#     modehb_nsgav2_path,modehb_epsnetv3_path,modehb_epsnetv2_path
# ]
# max_hv_contribution = get_max_hv(path)
#
# axes = plot_aggregated_hypervolume_nas(_get_experiments(shemoa_path, FILE_EXTENSION),
#                                         axes, 'EASH',
#                                         maxhv=max_hv_contribution)
#
# axes = plot_aggregated_hypervolume_nas(_get_experiments(mobohb_path, FILE_EXTENSION),
#                                         axes, 'MOBOHB',
#                                         maxhv=max_hv_contribution)
#
# axes = plot_aggregated_hypervolume_nas(_get_experiments(rs_path, FILE_EXTENSION),
#                                         axes, 'RS',
#                                         maxhv=max_hv_contribution)
#
# axes = plot_aggregated_hypervolume_nas(_get_experiments(modehb_ndsv2_path, FILE_EXTENSION),
#                                         axes, 'MODEHB_NDS_V2',
#                                         maxhv=max_hv_contribution)
# axes = plot_aggregated_hypervolume_nas(_get_experiments(modehb_epsnetv2_path, FILE_EXTENSION),
#                                         axes, 'MODEHB_EPSNET_V2',
#                                         maxhv=max_hv_contribution)

# axes = plot_aggregated_hypervolume_nas(_get_experiments(modehb_epsnetv3_path, FILE_EXTENSION),
#                                         axes, 'MODEHB_EPSNET_V3',
#                                         maxhv=max_hv_contribution)
# axes = plot_aggregated_hypervolume_nas(_get_experiments(modehb_nsgav2_path, FILE_EXTENSION),
#                                         axes, 'MODEHB_NSGA_V2',
#                                         maxhv=max_hv_contribution)
#
# axes = plot_aggregated_hypervolume_nas(_get_experiments(modehb_nsgav3_path, FILE_EXTENSION),
#                                         axes, 'MODEHB_NSGA_V3',
#                                         maxhv=max_hv_contribution)
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\nas\\modehb_nas_hvv3'
# axes = plot_aggregated_hypervolume_nas(_get_experiments(path, FILE_EXTENSION),
#                                         axes, 'MODEHB_HVV3',
#                                         maxhv=MAX_HV_CONTRIBUTION)



#plt.savefig(f'aggregated_time_mean.pdf', dpi=450)

############### ADULT ##########################################

# fig, axes = plt.subplots(1, figsize=(10, 5))
# FILE_EXTENSION = 'csv'
# X_LIMIT=3600
#
# rs_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\adult\\adult_rs'
# modehb_path ='C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\adult\\adult_modehb'
# mobohb_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\adult\\adult_mobohb'
# shemoa_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\adult\\adult_shemoa'
# modehb_epsnetv2_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\adult_epsnet'
# #
# modehb_epsnetv3_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\adult_epsnet_v3'
# #
# # modehb_epsnetv4_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\adult_epsnet_v4'
#
# modehb_nsgav2_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\adult_nsga_v2'
#
# modehb_nsgav3_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\adult_nsga_v3'
# #
# # modehb_nsgav4_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\adult_nsga_v4'
#
# path=[
#     rs_path, mobohb_path, modehb_path, shemoa_path, modehb_epsnetv2_path,modehb_nsgav2_path
#           ,modehb_epsnetv3_path,modehb_nsgav3_path
#     # modehb_nsgav4_path,,modehb_epsnetv4_path
# ]
# max_hv_contribution = get_max_hv(path)
# print("max hv contri for adult :",max_hv_contribution)
# # axes = plot_aggregated_norm_hypervolume(_get_experiments(rs_path, FILE_EXTENSION), axes, 'RS',
# #                                         max_hv_contribution,x_limit=X_LIMIT)
# #
# # axes = plot_aggregated_norm_hypervolume(_get_experiments(mobohb_path, FILE_EXTENSION), axes, 'MOBOHB',
# #                                         maxhv=max_hv_contribution,x_limit=X_LIMIT)
# # axes = plot_aggregated_norm_hypervolume(_get_experiments(shemoa_path, FILE_EXTENSION),  axes, 'EASH',
# #                                         maxhv=max_hv_contribution,x_limit=X_LIMIT)
# axes = plot_aggregated_norm_hypervolume(_get_experiments(modehb_epsnetv2_path, FILE_EXTENSION), axes, 'MODEHB_EPSNET_V2',
#                                         maxhv=max_hv_contribution,x_limit=X_LIMIT)
# axes = plot_aggregated_norm_hypervolume(_get_experiments(modehb_epsnetv3_path, FILE_EXTENSION), axes, 'MODEHB_EPSNET_V3',
#                                         maxhv=max_hv_contribution,x_limit=X_LIMIT)
# axes = plot_aggregated_norm_hypervolume(_get_experiments(modehb_nsgav2_path, FILE_EXTENSION), axes, 'MODEHB_NSGA_V2',
#                                         maxhv=max_hv_contribution,x_limit=X_LIMIT)
# axes = plot_aggregated_norm_hypervolume(_get_experiments(modehb_nsgav3_path, FILE_EXTENSION), axes, 'MODEHB_NSGA_V3',
#                                         maxhv=max_hv_contribution,x_limit=X_LIMIT)
# # axes = plot_aggregated_norm_hypervolume(_get_experiments(modehb_nsgav4_path, FILE_EXTENSION), axes, 'MODEHB_NSGA_V4',
# #                                         maxhv=max_hv_contribution,x_limit=X_LIMIT)
# # axes = plot_aggregated_norm_hypervolume(_get_experiments(modehb_epsnetv4_path, FILE_EXTENSION), axes, 'MODEHB_EPSNET_V4',
# #                                         maxhv=max_hv_contribution,x_limit=X_LIMIT)
# plt.savefig(f'aggregated_time_mean.pdf', dpi=450)




############### WIKI ##########################################

#
# fig, axes = plt.subplots(1, figsize=(10, 5))
# FILE_EXTENSION = 'csv'
# X_LIMIT=43200
# rs_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\wiki\\wiki_rs'
# modehb_path ='C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\wiki\\wiki_modehb'
# mobohb_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\wiki\\wiki_mobohb1'
# shemoa_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\wiki\\wiki_shemoa'
# modehb_epsnetv2_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\wiki_epsnet'
# modehb_epsnetv3_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\wiki_epsnet_v3'
# modehb_epsnetv4_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\wiki_epsnet_v4'
# modehb_nsgav2_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\wiki_nsga_v2'
# modehb_nsgav3_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\wiki_nsga_v3'
# modehb_nsgav4_path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\wiki_nsga_v4'
# ref_point = [-10.0, -1.0]
# path=[
#     rs_path, mobohb_path, modehb_path, shemoa_path, modehb_epsnetv2_path,modehb_epsnetv3_path,
#     modehb_epsnetv4_path,modehb_nsgav2_path,modehb_nsgav3_path,modehb_nsgav4_path
# ]
# max_hv_contribution = get_max_hv(path, ref_point)
#
#
# axes = plot_aggregated_norm_hypervolume(_get_experiments(rs_path, FILE_EXTENSION),
#                                         axes, 'RS',
#                                         maxhv=max_hv_contribution,x_limit=X_LIMIT)
#
# # axes = plot_aggregated_norm_hypervolume(_get_experiments(modehb_path, FILE_EXTENSION), axes, 'MODEHB_NDS_V2',
# #                                         maxhv=max_hv_contribution,x_limit=X_LIMIT)
# #
# axes = plot_aggregated_norm_hypervolume(_get_experiments(mobohb_path, FILE_EXTENSION), axes, 'MOBOHB',
#                                         maxhv=max_hv_contribution,x_limit=X_LIMIT)
#
# axes = plot_aggregated_norm_hypervolume(_get_experiments(shemoa_path, FILE_EXTENSION),  axes, 'EASH',
#                                         maxhv=max_hv_contribution,x_limit=X_LIMIT)
#
# axes = plot_aggregated_norm_hypervolume(_get_experiments(modehb_epsnetv2_path, FILE_EXTENSION), axes, 'MODEHB_EPSNET_V2',
#                                         maxhv=max_hv_contribution,x_limit=X_LIMIT)
#
# # axes = plot_aggregated_norm_hypervolume(_get_experiments(modehb_epsnetv3_path, FILE_EXTENSION), axes, 'MODEHB_EPSNET_V3',
# #                                         maxhv=max_hv_contribution,x_limit=X_LIMIT)
#
# axes = plot_aggregated_norm_hypervolume(_get_experiments(modehb_nsgav2_path, FILE_EXTENSION), axes, 'MODEHB_NSGA_V2',
#                                         maxhv=max_hv_contribution,x_limit=X_LIMIT)
#
# # axes = plot_aggregated_norm_hypervolume(_get_experiments(modehb_nsgav3_path, FILE_EXTENSION), axes, 'MODEHB_NSGA_V3',
# #                                         maxhv=max_hv_contribution,x_limit=X_LIMIT)
# #
# # axes = plot_aggregated_norm_hypervolume(_get_experiments(modehb_nsgav4_path, FILE_EXTENSION), axes, 'MODEHB_NSGA_V4',
# #                                         maxhv=max_hv_contribution,x_limit=X_LIMIT)
#
#
#
# plt.savefig(f'aggregated_time_mean.pdf', dpi=450)
#
