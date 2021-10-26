from dependencies.multiobjbaselines.baselines.problems import get_fashion
from dependencies.multiobjbaselines.baselines.problems.fashion import FashionSearchSpace
import DEHB.dehb.optimizers.modehb as mo
import sys
import argparse
from loguru import logger
import os
import time
import json
from dependencies.multiobjbaselines.baselines.problems.flowers.flowernet import evaluate_network

logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}

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

args = input_arguments()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False
min_budget = args.min_budget
max_budget = args.max_budget
dataset = args.dataset

output_path = create_output_dir(args)
output_path = args.output_path + "_" + str(args.run_id) + '/'
os.makedirs(output_path, exist_ok=True)

dimensions = len(FashionSearchSpace.get_hyperparameters())

def objective_function(cfg, seed, budget, run=1, **kwargs):
    start = time.time()
    metrics = evaluate_network(cfg,budget=budget)
    acc = metrics['val_acc_1']
    total_model_params = metrics['n_params']
    logger.info("budget:{}",budget)
    with open(output_path + 'dehb_run.json', 'a+')as f:
        json.dump({'configuration': dict(cfg), 'error': acc, 'top3': 1 - acc,
                   'n_params': total_model_params, 'num_epochs': budget}, f)

        f.write("\n")

    return ({"cost": time.time()-start,
             "fitness": [total_model_params, acc]})  # Because minimize!


modehb = mo.MODEHB(f=objective_function,
                   cs=FashionSearchSpace,
                   dimensions=dimensions,
                   min_budget=args.min_budget,
                   max_budget=args.max_budget,
                   eta=args.eta,
                   output_path=output_path,
                   # if client is not None and of type Client, n_workers is ignored
                   # if client is None, a Dask client with n_workers is set up
                   n_workers=args.n_workers,
                   seed=args.seed,
                   ref_point=[1, 1])