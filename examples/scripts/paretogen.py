import glob
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
import os
import numpy as np

def get_metrices(path):
    extension = 'txt'
    os.chdir(path)
    result = glob.glob('sort*.{}'.format(extension))
    data = []
    for f in result:
        exp_data = np.loadtxt(f)
        if (len(exp_data.shape) == 1):
            exp_data = np.array([exp_data])
        data.extend(exp_data.tolist())
    return data

def plot_pareto_fronts(
        #experiments: List[Experiment],
        data,
        ax,
        exp_type,
        xl,
        yl
):
        values_x = data[:,0]
        values_y = data[:,1]

        values = np.asarray([values_x, values_y]).T
        values = values[is_non_dominated(torch.as_tensor(values))]
        values = values[values[:, 0].argsort()]
        print("pareto:",values,"exp",exp_type)
        ax.plot(values[:, 0], values[:, 1], '-o', lw=3.0, label=exp_type)

        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend()
        return ax


############### ADULT ###########################################
# fig, axes = plt.subplots(1, figsize=(10, 5))
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\adult_nsga_v2'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='MODEHB_NSGA_V2',xl='val-error',yl='dsp')
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\adult_epsnet'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='MODEHB_EPSNET_V2',xl='val-error',yl='dsp')
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\adult\\adult_rs'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='RS',xl='val-error',yl='dsp')
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\adult\\adult_shemoa'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='EASH',xl='val-error',yl='dsp')
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\adult\\adult_mobohb'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='MOBOHB',xl='val-error',yl='dsp')
#
# plt.savefig(f'../aggregated_pareto.pdf', dpi=450)
#


##################################################FLOWER###########################################################3
# fig, axes = plt.subplots(1, figsize=(10, 5))
# X_LABEL="val-acc"
# Y_LABEL = "Log(num_params)"
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_modehb\\pc_nds_sv2\\experiments\\metrices'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='MODEHB_NSGA_V2',xl=X_LABEL,yl=Y_LABEL)
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\flower_epsnet'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='MODEHB_EPSNET_V2',xl='val-error',yl='dsp')
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_rs\\flower_rs_experiments'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='RS',xl=X_LABEL,yl=Y_LABEL)
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_shemoa\\flower_shemoa_experiments'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='EASH',xl=X_LABEL,yl=Y_LABEL)
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_mobohb\\flower_mobohb_experiments'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='MOBOHB',xl=X_LABEL,yl=Y_LABEL)
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_mosh\\experiments'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='MOSH',xl=X_LABEL,yl=Y_LABEL)
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\flower_runs\\flower_msehvi\\experiments'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='MSEHVI',xl='val-error',yl='dsp')
#
# plt.savefig(f'../aggregated_pareto.pdf', dpi=450)


############################################################FASHION ##################################################
#
fig, axes = plt.subplots(1, figsize=(10, 5))
X_LABEL="val-acc"
Y_LABEL = "Log(num_params)"

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\fashion_nsga_v2'
data = get_metrices(path)
axes = plot_pareto_fronts(np.array(data),axes,exp_type='MODEHB_NSGA_V2',xl=X_LABEL,yl=Y_LABEL)

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\fashion_epsnet_v2'
data = get_metrices(path)
axes = plot_pareto_fronts(np.array(data),axes,exp_type='MODEHB_EPSNET_V2',xl=X_LABEL,yl=Y_LABEL)

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\fashion_runs\\fashion_rs\\fashion_rs_experiments'
data = get_metrices(path)
axes = plot_pareto_fronts(np.array(data),axes,exp_type='RS',xl=X_LABEL,yl=Y_LABEL)

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\fashion_runs\\fashion_msehvi\\fashion_msehvi_experiments'
data = get_metrices(path)
axes = plot_pareto_fronts(np.array(data),axes,exp_type='MSEHVI',xl=X_LABEL,yl=Y_LABEL)
#plt.savefig(f'../aggregated_pareto.pdf', dpi=450)

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\fashion_runs\\fashion_mobohb\\fashion_mobohb_experiments'
data = get_metrices(path)
axes = plot_pareto_fronts(np.array(data),axes,exp_type='MOBOHB',xl=X_LABEL,yl=Y_LABEL)

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\fashion_runs\\fashion_moshbananas\\fashion_moshbananas_experiments'
data = get_metrices(path)
axes = plot_pareto_fronts(np.array(data),axes,exp_type='MOSH',xl=X_LABEL,yl=Y_LABEL)

path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\fashion_runs\\fashion_shemoa\\fashion_shemoa_experiments'
data = get_metrices(path)
axes = plot_pareto_fronts(np.array(data),axes,exp_type='EASH',xl=X_LABEL,yl=Y_LABEL)

plt.savefig(f'aggregated_pareto.pdf', dpi=450)

#

####################################### WIKI ######################################################################

# fig, axes = plt.subplots(1, figsize=(10, 5))
# X_LABEL="log perplexity"
# Y_LABEL = "val error"
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\wiki_nsga_v2'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='MODEHB_NSGA_V2',xl=X_LABEL,yl=Y_LABEL)
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\wiki\\wiki_rs'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='RS',xl=X_LABEL,yl=Y_LABEL)
#
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\wiki\\wiki_mobohb1'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='MOBOHB',xl=X_LABEL,yl=Y_LABEL)
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\wiki\\wiki_shemoa'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='EASH',xl=X_LABEL,yl=Y_LABEL)
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\wiki_epsnet'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='MODEHB_EPSNET_V2',xl=X_LABEL,yl=Y_LABEL)
# plt.savefig(f'aggregated_pareto.pdf', dpi=450)


########################################################### NAS #######################################################
# fig, axes = plt.subplots(1, figsize=(10, 5))
# X_LABEL="val-err"
# Y_LABEL = "norm prediction time"
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\nas_nsga_v2'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='MODEHB_NSGA_V2',xl=X_LABEL,yl=Y_LABEL)
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\nas_epsnet'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='MODEHB_EPSNET_V2',xl=X_LABEL,yl=Y_LABEL)
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\nas\\rs_nas'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='RS',xl=X_LABEL,yl=Y_LABEL)
#
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\nas\\mobohb_nas1'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='MOBOHB',xl=X_LABEL,yl=Y_LABEL)
#
# path = 'C:\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\runs\\nas\\shemoa_nas1'
# data = get_metrices(path)
# axes = plot_pareto_fronts(np.array(data),axes,exp_type='EASH',xl=X_LABEL,yl=Y_LABEL)
#
# plt.savefig(f'aggregated_pareto.pdf', dpi=450)