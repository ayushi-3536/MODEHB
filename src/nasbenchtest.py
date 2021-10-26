''' Runs MODEHB on NAS-Bench-201 '''

import os
import sys
from nats_bench import create

sys.path.append(os.path.join(os.getcwd(), '../nas201/'))
sys.path.append(os.path.join(os.getcwd(), '../AutoDL-Projects/lib/'))
import json
import pickle
import argparse
import numpy as np
import ConfigSpace
from loguru import logger
from dependencies.XAutoDL.xautodl.models import CellStructure, get_search_spaces
import DEHB.dehb.optimizers.modehb as mo

logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}


# From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/BOHB.py
## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
def get_configuration_space(max_nodes, search_space):
    cs = ConfigSpace.ConfigurationSpace()
    # edge2index   = {}
    for i in range(1, max_nodes):
        for j in range(i):
            node_str = '{:}<-{:}'.format(i, j)
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(node_str, search_space))
    return cs


# From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/BOHB.py
## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
def config2structure_func(max_nodes):
    def config2structure(config):
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                op_name = config[node_str]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    return config2structure


# def calculate_regrets(history, runtime):
#     assert len(runtime) == len(history)
#     global dataset, api, de, max_budget
#
#     regret_test = []
#     regret_validation = []
#     inc = np.inf
#     test_regret = 1
#     validation_regret = 1
#     for i in range(len(history)):
#         config, valid_regret, budget = history[i]
#         valid_regret = valid_regret - y_star_valid
#         if valid_regret <= inc:
#             inc = valid_regret
#             config = de.vector_to_configspace(config)
#             structure = config2structure(config)
#             arch_index = api.query_index_by_arch(structure)
#             info = api.get_more_info(arch_index, dataset, max_budget, False, False)
#             test_regret = (1 - (info['test-accuracy'] / 100)) - y_star_test
#         regret_validation.append(inc)
#         regret_test.append(test_regret)
#     res = {}
#     res['regret_test'] = regret_test
#     res['regret_validation'] = regret_validation
#     res['runtime'] = np.cumsum(runtime).tolist()
#     return res


def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(output_path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()


def find_nas201_best(api, dataset):
    arch, y_star_test = api.find_best(dataset=dataset, metric_on_set='ori-test')
    _, y_star_valid = api.find_best(dataset=dataset, metric_on_set='x-valid')
    return 1 - (y_star_valid / 100), 1 - (y_star_test / 100)


def input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix_seed', default='False', type=str, choices=['True', 'False'],
                        nargs='?', help='seed')
    parser.add_argument('--run_id', default=1, type=int, nargs='?',
                        help='unique number to identify this run')
    parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
    parser.add_argument('--run_start', default=0, type=int, nargs='?',
                        help='run index to start with for multiple runs')
    parser.add_argument('--dataset', default='cifar10-valid', type=str, nargs='?',
                        choices=['cifar10-valid', 'cifar100', 'ImageNet16-120'],
                        help='choose the dataset')
    parser.add_argument('--max_nodes', default=4, type=int, nargs='?',
                        help='maximum number of nodes in the cell')
    parser.add_argument('--iter', default=100, type=int, nargs='?',
                        help='number of DEHB iterations')
    parser.add_argument('--gens', default=1, type=int, nargs='?',
                        help='number of generations for DE to evolve')
    parser.add_argument('--output_path', default="/content/drive/MyDrive/run/changedk/30_runtime/", type=str, nargs='?',
                        help='specifies the path where the results will be saved')
    strategy_choices = ['rand1_bin', 'rand2_bin', 'rand2dir_bin', 'best1_bin', 'best2_bin',
                        'currenttobest1_bin', 'randtobest1_bin',
                        'rand1_exp', 'rand2_exp', 'rand2dir_exp', 'best1_exp', 'best2_exp',
                        'currenttobest1_exp', 'randtobest1_exp']
    parser.add_argument('--strategy', default="rand1_bin", choices=strategy_choices,
                        type=str, nargs='?',
                        help="specify the DE strategy from among {}".format(strategy_choices))
    parser.add_argument('--mutation_factor', default=0.5, type=float, nargs='?',
                        help='mutation factor value')
    parser.add_argument('--crossover_prob', default=0.5, type=float, nargs='?',
                        help='probability of crossover')
    parser.add_argument('--boundary_fix_type', default='random', type=str, nargs='?',
                        help="strategy to handle solutions outside range {'random', 'clip'}")
    parser.add_argument('--min_budget', default=11, type=int, nargs='?',
                        help='minimum budget')
    parser.add_argument('--max_budget', default=199, type=int, nargs='?',
                        help='maximum budget')
    parser.add_argument('--eta', default=3, type=int, nargs='?',
                        help='hyperband eta')
    parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                        help='to print progress or not')
    parser.add_argument('--folder', default=None, type=str, nargs='?',
                        help='name of folder where files will be dumped')
    parser.add_argument('--nas_bench_file', default='..//NATS-tss-v1_0-3ffb9-simple', type=str, nargs='?',
                        help='location of nas benchmark file')
    parser.add_argument('--version', default=None, type=str, nargs='?',
                        help='DEHB version to run')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of CPU workers for DEHB to distribute function evaluations to')

    parser.add_argument('-s', "--constraint_max_model_size",
                        default=2e7,
                        help="maximal model size constraint",
                        type=int)
    parser.add_argument('-p', "--constraint_min_precision",
                        default=0.42,
                        help='minimal constraint constraint',
                        type=float)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 123)')

    args = parser.parse_args()
    return args

time = 0
# Custom objective function for DEHB to interface NASBench-201
def nas_query_function(cfg, seed=1, budget=200, run=1, **kwargs):
    global dataset, api, time
    structure = config2structure(cfg)
    arch_index = api.query_index_by_arch(structure)
    if budget is not None:
        budget = int(budget)
    # From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/R_EA.py
    ## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
    info = api.get_more_info(arch_index, dataset, iepoch=budget, hp="200", is_random=True)
    logger.info("info:{}", info)
    info_cost = api.get_cost_info(arch_index, dataset)
    logger.info("info:{}", info)
    # validation_accuracy, latency, time_cost, current_total_time_cost = api.simulate_train_eval(arch_index,
    #                                                                                            dataset=dataset,
    #                                                                                            iepoch=budget, hp='200')
    # logger.info("simulated val acc:{},latency:{},tc:{},ttc:{}", validation_accuracy, latency, time_cost,
    #             current_total_time_cost)

    # a dict of all trials for 1st net on cifar100, where the key is the seed
    # results = api.query_by_index(arch_index,dataset)
    # logger.info('There are {:} trials for this architecture [{:}] on {}'.format(len(results), api[1],dataset))
    try:
        fitness = info['valid-accuracy']
    except:
        fitness = info['valtest-accuracy']
    runtime = info['train-all-time']
    try:
        #cost = info['valid-per-time']
        params= info_cost['params']
    except:
        #cost = info['valtest-per-time']
        params= 1
    valid_acc = fitness
    error = 1 - fitness / 100
    with open(output_path + 'dehb_run.json', 'a+')as f:
        json.dump({'configuration': dict(cfg), 'error': error, 'acc': valid_acc,
                   'params': params, 'num_epochs': budget}, f)

        f.write("\n")
    time += (runtime+info['valid-per-time'])
    logger.info("time exhaused:{}",time)
    # return fitness, cost
    return ({"cost":time ,
             "fitness": [-fitness, params]})


# Initializing DEHB object

def call_optimizer(args, cs, output_path, dimensions):
    modehb = mo.MODEHB(f=nas_query_function,
                       cs=cs,
                       dimensions=dimensions,
                       min_budget=args.min_budget,
                       max_budget=args.max_budget,
                       eta=args.eta,
                       constraint_model_size=args.constraint_max_model_size,
                       constraint_min_precision=args.constraint_min_precision,
                       output_path=output_path,
                       # if client is not None and of type Client, n_workers is ignored
                       # if client is None, a Dask client with n_workers is set up
                       n_workers=args.n_workers,
                       seed=args.seed,
                       ref_point=[1, 1])

    if args.runs is None:  # for a single run
        if not args.fix_seed:
            np.random.seed(args.run_id)
        # Running DE iterations
        runtime, history, pareto_pop, pareto_fit = modehb.run(iterations=args.iter, verbose=args.verbose)
        # res = calculate_regrets(history, runtime)

    else:  # for multiple runs
        for run_id, _ in enumerate(range(args.runs), start=args.run_start):
            if not args.fix_seed:
                np.random.seed(run_id)
            if args.verbose:
                print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
            # Running DE iterations
            runtime, history, pareto_pop, pareto_fit = modehb.run(iterations=args.iter,
                                                                  verbose=args.verbose)
            # res = calculate_regrets(history, runtime)
            # essential step to not accumulate consecutive runs
            modehb.reset()

    save_configspace(cs, output_path)


def load_nas_201_api(file):
    api = create(file, 'tss', fast_mode=True, verbose=False)
    search_space = get_search_spaces('cell', 'nas-bench-201')
    return api, search_space


def create_output_dir(args):
    # Directory where files will be written
    if args.folder is None:
        folder = "dehb"
        if args.version is not None:
            folder = "dehb_v{}".format(args.version)
    else:
        folder = args.folder

    output_path = os.path.join(args.output_path, args.dataset, folder)
    return output_path


args = input_arguments()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False
min_budget = args.min_budget
max_budget = args.max_budget
dataset = args.dataset

output_path = create_output_dir(args)
output_path = args.output_path + "_" + str(args.run_id) + '/'
os.makedirs(output_path, exist_ok=True)
api, search_space = load_nas_201_api(args.nas_bench_file)

# Parameter space to be used by DE
cs = get_configuration_space(args.max_nodes, search_space)
dimensions = len(cs.get_hyperparameters())
config2structure = config2structure_func(args.max_nodes)
# y_star_valid, y_star_test = find_nas201_best(api, dataset)
# inc_config = cs.get_default_configuration().get_array().tolist()
call_optimizer(args, cs, output_path, dimensions)
