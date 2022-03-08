from .problems.flowers import FlowersSearchSpace
from .problems.fashion import FashionSearchSpace
import sys
from loguru import logger
import time
from .default_utils import *
import json
import os
from .problems.flowers import evaluate_network as flowernet
from .problems.fashion import evaluate_network as fashionnet

import numpy as np

logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}


def input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix_seed', default='False', type=str, choices=['True', 'False'],
                        nargs='?', help='seed')
    parser.add_argument('--run_id', default=1, type=int, nargs='?',
                        help='unique number to identify this run')
    parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
    parser.add_argument('--dataset', default='cifar10-valid', type=str, nargs='?',
                        choices=['cifar10-valid', 'cifar100', 'ImageNet16-120'],
                        help='choose the dataset')
    parser.add_argument('--max_nodes', default=4, type=int, nargs='?',
                        help='maximum number of nodes in the cell')
    parser.add_argument('--gens', default=1, type=int, nargs='?',
                        help='number of generations for DE to evolve')
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
    parser.add_argument('--min_budget', default=5, type=int, nargs='?',
                        help='minimum budget')
    parser.add_argument('--max_budget', default=25, type=int, nargs='?',
                        help='maximum budget')
    parser.add_argument('--eta', default=3, type=int, nargs='?',
                        help='hyperband eta')
    parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                        help='to print progress or not')
    parser.add_argument('--folder', default=None, type=str, nargs='?',
                        help='name of folder where files will be dumped')
    parser.add_argument('--version', default=None, type=str, nargs='?',
                        help='DEHB version to run')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of CPU workers for DEHB to distribute function evaluations to')
    parser.add_argument('--runtime', type=int, default=400,
                        help='total runtime')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='total runtime')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 123)')

    parser.add_argument('--output_path', default="/content/drive/MyDrive/run/rs_flower/", type=str, nargs='?',
                        help='specifies the path where the results will be saved')

    # add experiment specific arguments here
    args = parser.parse_args()
    return args


def objective_function(cfg, budget):
    start = time.time()
    metrics = fashionnet(cfg, budget=int(budget))
    acc = metrics['val_acc_1']
    cost = time.time() - start
    total_model_params = metrics['num_params']
    total_runtime = metrics['total_runtime']
    eval_runtime = metrics['eval_runtime']
    logger.info("budget:{}, numparams:{}, acc:{}", budget, total_model_params, acc)
    with open(output_path + 'dehb_run.json', 'a+')as f:
        json.dump({'configuration': dict(cfg), 'acc': acc,
                   'n_params': total_model_params, 'num_epochs': budget, 'cost': cost}, f)

        f.write("\n")

    return ({"cost": total_runtime,
             "fitness": [acc, total_model_params,eval_runtime]})  # Because minimize!


args = input_arguments()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False
output_path = create_output_dir(args)
output_path = args.output_path + "_" + str(args.seed) + '/'
os.makedirs(output_path, exist_ok=True)
cs = FlowersSearchSpace()


def rs(seed):
    np.random.seed(seed)
    cs.seed(seed)
    for i in range(20000):
        config = cs.sample_configuration()
        run_info = objective_function(config, args.max_budget)
        fitness, cost = run_info["fitness"], run_info["cost"]
        logger.info("fitness:{}", fitness)
        fit = [np.array([fitness[0], fitness[1],fitness[2],cost])]
        with open(os.path.join(output_path, "res_random_search{}.txt"), 'a+') as f:
            np.savetxt(f, fit)


rs(args.seed)
