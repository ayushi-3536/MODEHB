from .problems.fashion import FashionSearchSpace
from ..DEHB.dehb.optimizers import MODEHB
import sys
from loguru import logger
import time
import json
from .default_utils import *
from .problems.fashion.fashionnet import evaluate_network
import signal
logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}
class OutOfTimeException(Exception):
    # Custom exception for easy handling of timeout
    pass

def signal_handler(sig, frame):
    logger.info('Job is being cancelled')
    raise OutOfTimeException


def input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix_seed', default='False', type=str, choices=['True', 'False'],
                        nargs='?', help='seed')
    parser.add_argument('--run_id', default=1, type=int, nargs='?',
                        help='unique number to identify this run')
    parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')

    parser.add_argument('--max_nodes', default=4, type=int, nargs='?',
                        help='maximum number of nodes in the cell')
    parser.add_argument('--gens', default=1, type=int, nargs='?',
                        help='number of generations for DE to evolve')
    parser.add_argument('--output_path', default="./fashion_logs", type=str, nargs='?',
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
    parser.add_argument('--runtime', type=int, default=10800,
                        help='total runtime')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='total runtime')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 123)')

    args = parser.parse_args()
    return args


def objective_function(cfg, seed, budget, run=1, **kwargs):
    start = time.time()
    metrics = evaluate_network(cfg,budget=int(budget))
    acc = metrics['val_acc_1']
    cost = time.time()-start
    total_model_params = metrics['num_params']
    logger.info("budget:{}",budget)
    with open(output_path + 'dehb_run.json', 'a+')as f:
        json.dump({'configuration': dict(cfg), 'acc': acc,
                   'num_params': total_model_params, 'num_epochs': budget,'cost':cost}, f)

        f.write("\n")

    return ({"cost": cost,
             "fitness": [total_model_params, acc]})  # Because minimize!


args = input_arguments()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False
output_path = create_output_dir(args)
output_path = args.output_path + "_" + str(args.run_id) + '/'
os.makedirs(output_path, exist_ok=True)
cs = FashionSearchSpace()
signal.signal(signal.SIGALRM, signal_handler)  # register the handler
signal.alarm(args.runtime)
try:
    modehb = MODEHB(objective_function=objective_function,
                       cs=cs,
                       dimensions=len(cs.get_hyperparameters()),
                       min_budget=args.min_budget,
                       max_budget=args.max_budget,
                       eta=args.eta,
                       output_path=output_path,
                       # if client is not None and of type Client, n_workers is ignored
                       # if client is None, a Dask client with n_workers is set up
                       n_workers=args.n_workers,
                       seed=args.seed,
                       ref_point=[8, 0])
    modehb.run(total_cost=args.runtime)
except OutOfTimeException:
    logger.info("catching out of time error")
