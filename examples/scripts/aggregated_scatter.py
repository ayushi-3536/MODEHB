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


def make_mask(data_size, sample_size):
    print("sameple size",sample_size)

    print("remaining",data_size-sample_size)
    sample_size = int(sample_size)

    mask = np.array([True] * sample_size + [False ] * (data_size - sample_size))
    np.random.shuffle(mask)
    return mask


def plot_aggregated_scatter(
        aggregated_data,
        axes,
        x_label,
        y_label,
        experiments_names: List[str],
        # metric_x: Optional[str] = None,
        # metric_y: Optional[str] = None,
        # bound_max_x: Optional[float] = None,
        # bound_max_y: Optional[float] = None,
):

    # experiments = [
    #     [
    #         load_experiment(e) for e in glob.glob(f'{path}/{exp_name}_*_final.pickle')
    #     ] for exp_name in experiments_names
    # ]
    #
    # metrics = experiments[0][0].optimization_config.objective.metrics
    #
    # metric_x = metrics[0] if metric_x is None else next(m for m in metrics if m.name == metric_x)
    # metric_y = metrics[1] if metric_y is None else next(m for m in metrics if m.name == metric_y)

    fig, axes = plt.subplots(1, len(aggregated_data), figsize=(5 * len(aggregated_data), len(aggregated_data)))
    setlabel = True
    for i,(ax, data) in enumerate(zip(axes, aggregated_data)):
    #
    #     for i, experiment in enumerate(experiment_list[:5]):

            # data = experiment.fetch_data().df
            values_x = data[:,0]#data[data['metric_name'] == metric_x.name]['mean'].values
            values_y = data[:,1]#data[data['metric_name'] == metric_y.name]['mean'].values
            #
            # values_x = values_x if metric_x.lower_is_better else -1 * values_x
            # values_y = values_y if metric_y.lower_is_better else -1 * values_y

            num_elements = values_x.size

            iterations = np.arange(num_elements)

            # if i == 0:
            #     ax.set_title(experiment.name.split('_')[0])
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(experiments_names[i])


            mask = make_mask(len(values_x), 0.1*(len(values_x)))
            ax.scatter(values_x[mask], values_y[mask], c=iterations[mask], alpha=0.8)
            iterations = np.arange(values_x[mask].size)
            print("iteration min",iterations.min())
            print("iteration max",iterations.max())
            norm = plt.Normalize(iterations.min(), iterations.max())
            sm = ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('viridis'))

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.15)
            fig.colorbar(sm, cax=cax, label='Iterations')

    return axes



def get_scatter_plot(path):
        extension = 'txt'
        os.chdir(path)
        result = glob.glob('metric*.{}'.format(extension))
        # print(result)
        data = []
        for f in result:
                data.extend(np.loadtxt(f))
        print(np.array(data))

        return np.array(data)

import os
import numpy as np
###############FLOWER ###########################################
fig, axes = plt.subplots(1, figsize=(10, 5))
X_LABEL='val-error'
Y_LABEL='prediction time'



experiments=[]
path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\adult_nsga_v2'
experiments.append(get_scatter_plot(path))

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\adult_epsnet'
experiments.append(get_scatter_plot(path))

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\adult\\adult_rs'
experiments.append(get_scatter_plot(path))

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\adult\\adult_shemoa'
experiments.append(get_scatter_plot(path))

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\adult\\adult_mobohb'
experiments.append(get_scatter_plot(path))


#
names = ['MODEHB_NSGA_V2','MODEHB_EPSNET_V2','RS','SHEMOA','MOBOHB']
axes = plot_aggregated_scatter(np.array(experiments), axes, x_label=X_LABEL, y_label=Y_LABEL,experiments_names=names)
plt.show()
plt.savefig(f'../aggregated_scatter.pdf', dpi=450)

# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\adult\\adult_rs'
# extension = 'txt'
# os.chdir(path)
# result = glob.glob('sort*.{}'.format(extension))
# #print(result)
# data=[]
# for f in result:
#     data.extend(np.loadtxt(f))
# #experiments = [np.loadtxt(f) for f in result]
# print(np.array(data))
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='RS',xl='val-error',yl='dsp')
# plt.savefig(f'../aggregated_pareto.pdf', dpi=450)
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\adult\\adult_shemoa'
# extension = 'txt'
# os.chdir(path)
# result = glob.glob('sort*.{}'.format(extension))
# #print(result)
# data=[]
# for f in result:
#     data.extend(np.loadtxt(f))
# #experiments = [np.loadtxt(f) for f in result]
# print(np.array(data))
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='EASH',xl='val-error',yl='dsp')
# plt.savefig(f'../aggregated_pareto.pdf', dpi=450)

# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\adult\\adult_mobohb'
# extension = 'txt'
# os.chdir(path)
# result = glob.glob('sort*.{}'.format(extension))
# #print(result)
# data=[]
# for f in result:
#     data.extend(np.loadtxt(f))
# #experiments = [np.loadtxt(f) for f in result]
# print(np.array(data))
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='MOBOHB',xl='val-error',yl='dsp')
# plt.savefig(f'../aggregated_pareto.pdf', dpi=450)



##################################################FLOWER###########################################################3
# fig, axes = plt.subplots(1, figsize=(10, 6))
# X_LABEL="val-acc"
# Y_LABEL = "Log(num_params)"
# experiments=[]
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_modehb\\pc_nds_sv2\\experiments\\metrices'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\flower_epsnet'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_rs\\flower_rs_experiments'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_shemoa\\flower_shemoa_experiments'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_mobohb\\flower_mobohb_experiments'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_mosh\\experiments'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_msehvi\\experiments'
# experiments.append(get_scatter_plot(path))
# #
# names = ['MODEHB_NSGA_V2','MODEHB_EPSNET_V2','RS','SHEMOA','MOBOHB','MOSHBANANA','MSEHVI']
# axes = plot_aggregated_scatter(np.array(experiments), axes, x_label=X_LABEL, y_label=Y_LABEL,experiments_names=names)
# plt.show()
# plt.savefig(f'../aggregated_scatter.pdf', dpi=450)


############################################################FASHION ##################################################

# fig, axes = plt.subplots(1, figsize=(10, 8))
# X_LABEL="val-acc"
# Y_LABEL = "Log(num_params)"
# experiments=[]
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\fashion_epsnet_v2'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\fashion_nsga_v2'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\fashion_runs\\fashion_rs\\fashion_rs_experiments'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\fashion_runs\\fashion_shemoa\\fashion_shemoa_experiments'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\fashion_runs\\fashion_mobohb\\fashion_mobohb_experiments'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\fashion_runs\\fashion_moshbananas\\fashion_moshbananas_experiments'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\fashion_runs\\fashion_msehvi\\fashion_msehvi_experiments'
# experiments.append(get_scatter_plot(path))
# #
# names = ['MODEHB_NSGA_V2','MODEHB_EPSNET_V2','RS','SHEMOA','MOBOHB','MOSHBANANA','MSEHVI']
# axes = plot_aggregated_scatter(np.array(experiments), axes, x_label=X_LABEL, y_label=Y_LABEL,experiments_names=names)
# plt.show()
# plt.savefig(f'../aggregated_scatter.pdf', dpi=450)

####################################### WIKI ######################################################################
#
# fig, axes = plt.subplots(1, figsize=(10, 5))
# X_LABEL="log perplexity"
# Y_LABEL = "val error"
#
# experiments=[]
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\wiki\\wiki_modehb'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\wiki_epsnet'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\wiki\\wiki_rs'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\wiki\\wiki_shemoa'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\wiki\\wiki_mobohb1'
# experiments.append(get_scatter_plot(path))
#
#
#
# names = ['MODEHB_NSGA_V2','MODEHB_EPSNET_V2','RS','SHEMOA','MOBOHB']
# axes = plot_aggregated_scatter(np.array(experiments), axes, x_label=X_LABEL, y_label=Y_LABEL,experiments_names=names)
# plt.show()
# plt.savefig(f'../aggregated_scatter.pdf', dpi=450)


# ########################################################### NAS #######################################################
# fig, axes = plt.subplots(1, figsize=(10, 5))
# X_LABEL="val-err"
# Y_LABEL = "norm prediction time"
#
#
# experiments=[]
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\nas_nsga_v2'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\nas_epsnet'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\nas\\rs_nas'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\nas\\shemoa_nas1'
# experiments.append(get_scatter_plot(path))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\nas\\mobohb_nas1'
# experiments.append(get_scatter_plot(path))
#
#
# #
# names = ['MODEHB_NSGA_V2','MODEHB_EPSNET_V2','RS','SHEMOA','MOBOHB']
# axes = plot_aggregated_scatter(np.array(experiments), axes, x_label=X_LABEL, y_label=Y_LABEL,experiments_names=names)
# plt.show()
# plt.savefig(f'../aggregated_scatter.pdf', dpi=450)
