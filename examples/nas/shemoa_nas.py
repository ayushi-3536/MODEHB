from baselines import save_experiment
from baselines.methods.shemoa_nas import SHEMOA_NAS
from baselines.methods.shemoa import Mutation, Recombination, ParentSelection
import argparse
import sys
from loguru import logger
from baselines.problems.nasbench201.nas_bench_201 import get_nasbench201_err_time
from baselines.problems.nasbench201.nas_bench_201 import NasBench201
from baselines.problems.nasbench201.nas_search_space import NASSearchSpace

logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}
class OutOfTimeException(Exception):
    # Custom exception for easy handling of timeout
    pass

def timeouthandler(signum, frame):
    print('The end is nigh')
    raise OutOfTimeException

parser = argparse.ArgumentParser()
parser.add_argument('--runtime', type=int, default=24*3600*10,
                    help='total runtime')
parser.add_argument('--timeout', default=85000, type=int, help='Timeout in sec. 0 -> no timeout')
parser.add_argument('--seed', type=int, default=500, metavar='S',
                    help='random seed (default: 123)')
parser.add_argument('--benchmark_file',
                    default='/work/dlclarge1/sharmaa-mulobjtest/NATS-tss-v1_0-3ffb9-simple', type=str,
                    help='Timeout in sec. 0 -> no timeout')
parser.add_argument('--dataset', default='ImageNet16-120', type=str, help='Timeout in sec. 0 -> no timeout')

args = parser.parse_args()

N_init = 10
min_budget = 5
max_budget = 199

max_function_evals = 1000

mutation_type = Mutation.UNIFORM
recombination_type = Recombination.UNIFORM
selection_type = ParentSelection.TOURNAMENT

nb = NasBench201(file=args.benchmark_file, dataset=args.dataset)
search_space = NASSearchSpace()
#experiment = get_nasbench201(name='SHEMOA_NAS_' + str(args.seed), nb=nb, search_space=search_space)
experiment = get_nasbench201_err_time(name='SHEMOA_NAS_TIMEERR_199' + str(args.seed), nb=nb, search_space=search_space)




if __name__ == '__main__':



    #################
    #### SH-EMOA ####
    #################
    # signal.signal(signal.SIGALRM, timeouthandler)  # register the handler
    # signal.alarm(args.timeout)
    try:
        ea = SHEMOA_NAS(
            search_space,
            experiment,
            N_init, min_budget, max_budget,
            mutation_type=mutation_type,
            recombination_type=recombination_type,
            selection_type=selection_type,
            total_number_of_function_evaluations=max_function_evals,
            max_time=args.timeout,
            seed=args.seed,
            bench=nb
        )
        ea.optimize()
        save_experiment(experiment, f'{experiment.name}.pickle')
    except OutOfTimeException:
        print("catching time out of exception")
        save_experiment(experiment, f'{experiment.name}.pickle')
