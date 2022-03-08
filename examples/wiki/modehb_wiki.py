from baselines.methods.modehb.DEHB.dehb.optimizers import MODEHB

from baselines.problems import get_wikitext_ppl_score
from baselines.problems.wikitext import WikiSearchSpace
from baselines import save_experiment
import sys
from copy import deepcopy
from loguru import logger
import time
import json
from ax import Data, GeneratorRun, Arm, Models
import uuid
from baselines.methods.modehb.DEHB.default_utils import *
import signal
from datetime import datetime
# from baselines.problems.nas_bench_201 import get_nasbench201
# from baselines.problems.nas_bench_201 import NasBench201
# from baselines.problems.nas_search_space import NASSearchSpace


logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
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
    save_experiment(experiment, f'{experiment.name}.pickle')
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

    mo_strategy_choices = ['EPSNET', 'NSGA-II', 'MAXHV']
    parser.add_argument('--mo_strategy', default="EPSNET", choices=mo_strategy_choices,
                        type=str, nargs='?',
                        help="specify the multiobjective  strategy from among {}".format(mo_strategy_choices))
    mo_selection_choices = ['V2', 'V3', 'V4']
    parser.add_argument('--mo_selection_strategy', default="V2", choices=mo_selection_choices,
                        type=str, nargs='?',
                        help="specify the multiobjective selection strategy from among {}".format(mo_selection_choices))

    parser.add_argument('--mutation_factor', default=0.5, type=float, nargs='?',
                        help='mutation factor value')
    parser.add_argument('--crossover_prob', default=0.5, type=float, nargs='?',
                        help='probability of crossover')
    parser.add_argument('--boundary_fix_type', default='random', type=str, nargs='?',
                        help="strategy to handle solutions outside range {'random', 'clip'}")
    parser.add_argument('--min_budget', default=1, type=int, nargs='?',
                        help='minimum budget')
    parser.add_argument('--max_budget', default=81, type=int, nargs='?',
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
    parser.add_argument('--runtime', type=int, default=43200,
                        help='total runtime')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='total runtime')
    parser.add_argument('--timeout', default=45000, type=int, help='Timeout in sec. 0 -> no timeout')
    parser.add_argument('--seed', type=int, default=500, metavar='S',
                        help='random seed (default: 123)')

    parser.add_argument('--dataset', default='ImageNet16-120', type=str, help='Timeout in sec. 0 -> no timeout')


    args = parser.parse_args()
    return args


args = input_arguments()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False
output_path = create_output_dir(args)
output_path = args.output_path + "_" + str(args.run_id) + '/'
os.makedirs(output_path, exist_ok=True)
#cs = FashionSearchSpace()
#
signal.signal(signal.SIGALRM, signal_handler)  # register the handler
signal.alarm(args.timeout)

_id = uuid.uuid4()
num_evals = 0


#Parameters Wiki
search_space = WikiSearchSpace()
experiment = get_wikitext_ppl_score('MODEHB_Wikitext_{}_{}_{}_mib_{}_mab_{}'.format(args.seed,args.mo_strategy,args.mo_selection_strategy,args.min_budget,args.max_budget))
th_list = experiment.optimization_config.objective_thresholds
ref_point = [th.bound for th in th_list]

def wiki_objective_function(cfg, budget, run=1, **kwargs):
    global num_evals

    logger.debug("all keys:{}",cfg.keys())
    logger.debug("cfg:{}",cfg)
    # if budget not in cfg.keys():
    #     cfg[budget] = []

    logger.debug("budget:{}", budget)
    logger.debug("config sampled:{}", cfg)

    random_ax = Models.SOBOL(search_space.as_ax_space()).gen(1).arms[0].parameters
    logger.debug("random:{}",random_ax)
    # conf = ConfigSpace.Configuration(self.configspace, job.kwargs["config"])
    # self.configs[budget].append(conf.get_array())

    # random_ax = deepcopy(cfg)
    # logger.info("random ax after deep copy:{}",random_ax)


    #########################CONVERTING CONFIGSPACE TO AX SPACE FOR TRIAL EVALUATION###############################################################################################
    params = deepcopy(random_ax)
    params['budget'] = int(budget)

    #
    params['lr_factor'] = cfg['lr_factor']
    params['lr'] = cfg['lr']
    params['batch_size'] = cfg['batch_size']
    params['emsize'] = cfg['emsize']
    params['dropout'] = cfg['dropout']
    params['clip'] = cfg['clip']
    # params['batch_size'] = cfg['batch_size']
    # params['batch_norm'] = bool(cfg['batch_norm'])
    # params['global_avg_pooling'] = bool(cfg['global_avg_pooling'])

    trial_name = '{}-{}'.format(_id,num_evals)
    params['id'] = trial_name
    params['id'] = str(args.seed)

    logger.debug("params:{}",params)
    start_time = time.time()

    trial = experiment.new_trial(GeneratorRun([Arm(params, name=str(args.seed))]))

    data = experiment.eval_trial(trial)
    cost=time.time()-start_time
    num_evals += 1
    ppl = float(data.df[data.df['metric_name'] == 'log_perplexity']['mean'])
    error = float(data.df[data.df['metric_name'] == 'val_error']['mean'])
    logger.debug("dsp:{}",ppl,"test_error:{}",error,"cost:{}",cost)
    with open(output_path + 'dehb_run_{}.json'.format(args.seed), 'a+')as f:
        json.dump({'configuration': dict(cfg), 'dsp': ppl,
                   'test_error': error, 'num_epochs': budget,'cost':cost}, f)

        f.write("\n")
    return ({"cost":cost,
             "fitness": [ppl, error]})


try:
    modehb = MODEHB(objective_function=wiki_objective_function,
                       cs=search_space,
                       dimensions=len(search_space.as_uniform_space().get_hyperparameters()),
                       min_budget=args.min_budget,
                       max_budget=args.max_budget,
                       eta=args.eta,
                       output_path=output_path,
                       # if client is not None and of type Client, n_workers is ignored
                       # if client is None, a Dask client with n_workers is set up
                       n_workers=args.n_workers,
                       ##ToDo: retrieve reference point from experiment
                       ref_point=ref_point,
                       seed = args.seed,
                       mo_strategy = args.mo_strategy,
                       mo_selection_strategy = args.mo_selection_strategy
                    )
    modehb.run(total_cost=args.runtime)
    save_experiment(experiment, f'{experiment.name}.pickle')
except OutOfTimeException:
    logger.info("catching out of time error")
