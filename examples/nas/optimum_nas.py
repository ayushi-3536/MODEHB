from ax import Models
import signal

import traceback
import argparse
import sys

import torch
from tqdm import tqdm
import csv
from datetime import datetime
from problems.nasbench201.nas_bench_201 import get_nasbench201_err_time
from problems.nasbench201.nas_bench_201 import NasBench201
from problems.nasbench201.nas_search_space import NASSearchSpace
from baselines import save_experiment
from ax import Experiment, MultiObjectiveOptimizationConfig
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from matplotlib.cm import ScalarMappable
import numpy as np
from collections import defaultdict

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



    for i in tqdm(range(1, data.shape[0]), desc=experiment.name):
        result['hypervolume'].append(hv.compute(data[:i][is_non_dominated(data[:i])]))
        try:
            result['walltime'].append(((experiment.trials[i - 1].time_completed - experiment.time_created).total_seconds()))
        except:
            result['walltime'].append(0)
        result['evaluations'].append(i)

    return pd.DataFrame(result)

parser = argparse.ArgumentParser()
parser.add_argument('--runtime', type=int, default=1000000,
                    help='total runtime')
parser.add_argument('--timeout', default=85000, type=int, help='Timeout in sec. 0 -> no timeout')
parser.add_argument('--seed', type=int, default=500, metavar='S',
                    help='random seed (default: 123)')
parser.add_argument('--benchmark_file',
                    default='C:\\users\\ayush\\OneDrive\\Pictures\\nasbench\\NATS-tss-v1_0-3ffb9-simple', type=str,
                    help='Timeout in sec. 0 -> no timeout')
parser.add_argument('--dataset', default='ImageNet16-120', type=str, help='Timeout in sec. 0 -> no timeout')

args = parser.parse_args()

# with open("all_evals_nas_imagenet.csv", "a+") as outfile:
#     outfile.write('val_acc,num_params,train_all_time,train_per_time,val_per_time,val_all_time,'
#                   'prediction_time,error'+'\n')

import json

import pandas as pd
if __name__ == '__main__':

    # signal.signal(signal.SIGALRM, signal_handler)  # register the handler
    # signal.alarm(args.timeout)
    ###############################
    #### Random Search for NAS ####
    ###############################
    # try:
    #    for i in range(15625):
    #        nb = NasBench201(file=args.benchmark_file, dataset=args.dataset)
    #        info = nb.eval_network(budget=199,arch_index=i)
    #        logger.info("info:{}",nb.eval_network(budget=199,arch_index=i))
    #        #
    #        s = []
    #        for k,v in info.items():
    #            print(k, ":", v[0])
    #            s.append(str(v[0]))
    #        s=','.join(s)+'\n'
    #        print(s)
    #        with open("all_evals_nas_imagenet.csv", "a+") as outfile:
    #                outfile.write(s)
    #
    #    logger.info("saving the results")
    # except (Exception, KeyboardInterrupt) as err:
    #    traceback.print_exc()
    #    logger.error("catching error and checkpointing the result:{}",err,exc_info=True)


    df = pd.read_csv('../all_evals_nas_imagenet.csv')
    print(df)
    data_err = df['error']
    data_prediction_time = df['prediction_time']
    max_prediction_time = data_prediction_time.max()
    print(max_prediction_time)
    data_prediction_time = data_prediction_time/max_prediction_time
    data = pd.concat([data_err,data_prediction_time], axis=1)
    print(data)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(np.ascontiguousarray(np.asarray(data)), device=device).float()
    ref_point = [-1.0, -1.0]
    hv = Hypervolume(torch.tensor(ref_point))
    pareto = data[is_non_dominated(data)]
    hypervolume = hv.compute(pareto)
    print("pareto:{}",pareto)
    print("hypervolume:{}",hypervolume)
    with open("../imagenet_metric.csv", "a+") as outfile:
        outfile.write("hypervolume:{}\n".format(hypervolume))
        outfile.write("ref_point:{}\n".format(ref_point))

