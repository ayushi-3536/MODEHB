import torch
import numpy as np
from ax import Metric
from ax.core.search_space import SearchSpace
from ax.core.objective import MultiObjective
from ax.core.parameter import ChoiceParameter, FixedParameter, ParameterType, RangeParameter
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from baselines import MultiObjectiveSimpleExperiment
import sys
from loguru import logger
import os
from nats_bench import create
from .nas_search_space import NASSearchSpace
from baselines.problems.dependencies.XAutoDL.xautodl.models import CellStructure, get_search_spaces
import ConfigSpace
from time import time

logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}


def get_nasbench201(nb,search_space, name=None):
    val_acc = Metric('val_acc', True)
    train_all_time = Metric('train_all_time', True)
    val_per_time = Metric('val_per_time', True)
    num_params = Metric('num_params', True)

    objective = MultiObjective([val_acc, num_params])
    thresholds = [
        ObjectiveThreshold(val_acc, 0.0),
        ObjectiveThreshold(num_params, 2.0)
    ]
    optimization_config = MultiObjectiveOptimizationConfig(
        objective=objective,
        objective_thresholds=thresholds
    )

    return MultiObjectiveSimpleExperiment(
        name=name,
        search_space=search_space.as_ax_space(),
        eval_function=nb.eval_network,
        optimization_config=optimization_config,
        extra_metrics=[train_all_time, val_per_time]
    )
MAX_IMAGENET_PREDICTION_TIME = 106
def get_nasbench201_err_time(nb,search_space, name=None):
    val_acc = Metric('val_acc', True)
    train_all_time = Metric('train_all_time', True)
    train_per_time = Metric('train_per_time', True)
    val_per_time = Metric('val_per_time', True)
    val_all_time = Metric('val_all_time', True)
    prediction_time = Metric('prediction_time', True)
    norm_prediction_time = Metric('norm_prediction_time', True)
    num_params = Metric('num_params', True)
    error = Metric('error', True)

    objective = MultiObjective([error, norm_prediction_time])
    thresholds = [
        ObjectiveThreshold(error, 1.0),
        ObjectiveThreshold(norm_prediction_time, 1.0)

    ]
    optimization_config = MultiObjectiveOptimizationConfig(
        objective=objective,
        objective_thresholds=thresholds
    )

    return MultiObjectiveSimpleExperiment(
        name=name,
        search_space=search_space.as_ax_space(),
        eval_function=nb.eval_network,
        optimization_config=optimization_config,
        extra_metrics=[train_all_time,train_per_time,val_all_time,val_per_time, num_params,val_acc,prediction_time]
    )


class NasBench201:
    def __init__(self, file, dataset):
        print(os.getcwd())
        self.api = create(file, 'tss', fast_mode=True, verbose=False)
        self.dataset = dataset
        self.search_space = get_search_spaces('cell', 'nas-bench-201')
        max_nodes = 4  # no. of nodes in  a cell
        self.last_ts = time()
        self.init_time=time()
        #self.config2structure = config2structure_func(max_nodes)

    def set_last_ts(self, x):
        self.last_ts = x


    def get_search_space(self):
        return self.search_space

    def get_last_ts(self, x):
        return self.last_ts

    def _x_to_info(self, x):
        ops = [
            'none',
            'skip_connect',
            'nor_conv_1x1',
            'nor_conv_3x3',
            'avg_pool_3x3'
        ]

        p1, p2, p3 = ops[x['p1']], ops[x['p2']], ops[x['p3']]
        p4, p5, p6 = ops[x['p4']], ops[x['p5']], ops[x['p6']]
        arch = f'|{p1}~0|+|{p2}~0|{p3}~1|+|{p4}~0|{p5}~1|{p6}~2|'
        return arch

    # From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/BOHB.py
    ## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
    def config2structure(self,config):
            genotypes = []
            for i in range(1, 4):
                xlist = []
                for j in range(i):
                    node_str = '{:}<-{:}'.format(i, j)
                    op_name = config[node_str]
                    xlist.append((op_name, j))
                genotypes.append(tuple(xlist))
            return CellStructure(genotypes)

            # return config2structure

    def eval_network(self, cfg, budget=None):

        #structure = self._x_to_info(cfg)
        structure = self.config2structure(cfg)
        arch_index = self.api.query_index_by_arch(structure)
        if budget is not None:
            budget = int(budget)
        else:
            budget = cfg['budget']
        print("cfg:{},budget:{}", cfg, budget)
        # From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/R_EA.py
        ## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
        info = self.api.get_more_info(arch_index, self.dataset, iepoch=budget, hp="200", is_random=True)
        info_cost = self.api.get_cost_info(arch_index, self.dataset)
        logger.info("info:{}", info)
        try:
            val_acc1 = info['valid-accuracy']
        except:
            val_acc1 = info['valtest-accuracy']
        try:
            # cost = info['valid-per-time']
            num_params = info_cost['params']
        except:
            # cost = info['valtest-per-time']
            num_params = 0.0

        train_all_runtime = info['train-all-time']
        train_per_runtime = info['train-per-time']
        try:
            val_per_runtime = info['valid-per-time']
            val_all_runtime = info['valid-all-time']
        except:
            val_per_runtime = info['valtest-per-time']
            val_all_runtime = info['valtest-all-time']
        prediction_time = train_per_runtime + val_per_runtime
        logger.debug("prediction time:{}",prediction_time)
        norm_prediction_time = prediction_time/MAX_IMAGENET_PREDICTION_TIME
        logger.debug("norm prediction time :{}",norm_prediction_time)
        #returning mean and sem as (mean,sem), since we have single value of metric sem is 0.0
        return {
            'val_acc': (-1 * val_acc1, 0.0),
            'num_params': (num_params, 0.0),
            'train_all_time': (train_all_runtime, 0.0),
            'train_per_time': (train_per_runtime, 0.0),
            'val_per_time': (val_per_runtime, 0.0),
            'val_all_time': (val_all_runtime, 0.0),
            'prediction_time': (prediction_time, 0.0),
            'norm_prediction_time': (norm_prediction_time, 0.0),
            'error': (1 - (val_acc1/100),0.0)
        }


# def eval_network(self, cfg=None, budget=None,arch_index=None):
#
#
#     # structure = self._x_to_info(cfg)
#     if (arch_index == None):
#         structure = self.config2structure(cfg)
#     arch_index = self.api.query_index_by_arch(structure)
#     if budget is not None:
#         budget = int(budget)
#     else:
#         budget = cfg['budget']
#     print("cfg:{},budget:{}", cfg, budget)
#     # From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/R_EA.py
#     ## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
#     info = self.api.get_more_info(arch_index, self.dataset, iepoch=budget, hp="200", is_random=True)
#     info_cost = self.api.get_cost_info(arch_index, self.dataset)
#     logger.info("info:{}", info)
#     try:
#         val_acc1 = info['valid-accuracy']
#     except:
#         val_acc1 = info['valtest-accuracy']
#     try:
#         # cost = info['valid-per-time']
#         num_params = info_cost['params']
#     except:
#         # cost = info['valtest-per-time']
#         num_params = 0.0
#
#     train_all_runtime = info['train-all-time']
#     train_per_runtime = info['train-per-time']
#     try:
#         val_per_runtime = info['valid-per-time']
#         val_all_runtime = info['valid-all-time']
#     except:
#         val_per_runtime = info['valtest-per-time']
#         val_all_runtime = info['valtest-all-time']
#     prediction_time = train_per_runtime + val_per_runtime
#     norm_prediction_time = prediction_time / MAX_IMAGENET_PREDICTION_TIME
#
#     return {
#         'val_acc': (-1 * val_acc1, 0.0),
#         'num_params': (num_params, 0.0),
#         'train_all_time': (train_all_runtime, 0.0),
#         'train_per_time': (train_per_runtime, 0.0),
#         'val_per_time': (val_per_runtime, 0.0),
#         'val_all_time': (val_all_runtime, 0.0),
#         'prediction_time': (prediction_time, 0.0),
#         'norm_prediction_time': (norm_prediction_time, 0.0),
#         'error': (1 - (val_acc1 / 100), 0.0)
#     }
# import torch
# import numpy as np
# from ax import Metric
# from ax.core.search_space import SearchSpace
# from ax.core.objective import MultiObjective
# from ax.core.parameter import ChoiceParameter, FixedParameter, ParameterType, RangeParameter
# from ax.core.outcome_constraint import ObjectiveThreshold
# from ax.core.optimization_config import MultiObjectiveOptimizationConfig
# from baselines import MultiObjectiveSimpleExperiment
# import sys
# from loguru import logger
# import os
# from nats_bench import create
# from ..dependencies.XAutoDL.xautodl.models import CellStructure, get_search_spaces
# import ConfigSpace
#
#
# logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
# _logger_props = {
#     "format": "{time} {level} {message}",
#     "enqueue": True,
#     "rotation": "500 MB"
# }
#
# def get_nasbench201(file, dataset, name=None):
#
#     val_acc = Metric('val_acc', True)
#     train_all_time = Metric('train_all_time', True)
#     val_per_time = Metric('val_per_time', True)
#     num_params = Metric('num_params', True)
#
#     objective = MultiObjective([val_acc, num_params])
#     thresholds = [
#         ObjectiveThreshold(val_acc, 0.0),
#         ObjectiveThreshold(num_params, 2.0)
#     ]
#     optimization_config = MultiObjectiveOptimizationConfig(
#         objective=objective,
#         objective_thresholds=thresholds
#     )
#
#
#
#     nb,search_space = NasBench201(file=file, dataset=dataset)
#
#     return MultiObjectiveSimpleExperiment(
#         name=name,
#         search_space=search_space,
#         eval_function=nb.eval_network,
#         optimization_config=optimization_config,
#         extra_metrics=[train_all_time, val_per_time]
#     )
#
# # From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/BOHB.py
# ## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
# def get_configuration_space(max_nodes, search_space):
#     cs = ConfigSpace.ConfigurationSpace()
#     # edge2index   = {}
#     for i in range(1, max_nodes):
#         for j in range(i):
#             node_str = '{:}<-{:}'.format(i, j)
#             cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(node_str, search_space))
#     return cs
#
#
# # From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/BOHB.py
# ## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
# def config2structure_func(max_nodes):
#     def config2structure(config):
#         genotypes = []
#         for i in range(1, max_nodes):
#             xlist = []
#             for j in range(i):
#                 node_str = '{:}<-{:}'.format(i, j)
#                 op_name = config[node_str]
#                 xlist.append((op_name, j))
#             genotypes.append(tuple(xlist))
#         return CellStructure(genotypes)
#
#     return config2structure
#
# class NasBench201:
#     def __init__(self,file,dataset):
#         print(os.getcwd())
#         self.api = create(file, 'tss', fast_mode=True, verbose=False)
#         self.dataset = dataset
#         self.search_space = get_search_spaces('cell', 'nas-bench-201')
#         max_nodes=4 #no. of nodes in  a cell
#         self.config2structure = config2structure_func(max_nodes)
#         return self, self.search_space
#
#
#     def eval_network(self,cfg,budget=None):
#             structure = self.config2structure(cfg)
#             arch_index = self.api.query_index_by_arch(structure)
#             if budget is not None:
#                 budget = int(budget)
#             # From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/R_EA.py
#             ## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
#             info = self.api.get_more_info(arch_index, self.dataset, iepoch=budget, hp="200", is_random=True)
#             logger.info("info:{}", info)
#             info_cost = self.api.get_cost_info(arch_index, self.dataset)
#             logger.info("info:{}", info)
#             try:
#                 val_acc1 = info['valid-accuracy']
#             except:
#                 val_acc1 = info['valtest-accuracy']
#             try:
#                 # cost = info['valid-per-time']
#                 num_params = info_cost['params']
#             except:
#                 # cost = info['valtest-per-time']
#                 num_params = 1
#
#             train_runtime = info['train-all-time']
#             val_runtime = info['valid-per-time']
#             return {
#                     'val_acc_1':  (-100.0 * val_acc1, 0.0),
#                     'num_params': (num_params, 0.0),
#                     'train_all_time': (train_runtime),
#                     'val_per_time': (val_runtime)
#                     }
#

