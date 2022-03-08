from baselines.methods.mobohb_nas.run_mobohb import get_MOBOHB
from baselines import save_experiment
import sys
import signal
import argparse
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

def signal_handler(sig, frame):
    save_experiment(experiment, f'{experiment.name}.pickle')
    logger.info('Job is being cancelled')
    raise OutOfTimeException

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=int, default=24*3600*10,
                        help='total runtime')
    parser.add_argument('--timeout', default=600, type=int, help='Timeout in sec. 0 -> no timeout')
    parser.add_argument('--seed', type=int, default=500, metavar='S',
                        help='random seed (default: 123)')
    parser.add_argument('--benchmark_file',
                        default='/work/dlclarge1/sharmaa-mulobjtest/NATS-tss-v1_0-3ffb9-simple', type=str,
                        help='Timeout in sec. 0 -> no timeout')
    parser.add_argument('--dataset', default='ImageNet16-120', type=str, help='Timeout in sec. 0 -> no timeout')

    args = parser.parse_args()

    # Parameters NAS
    N_init = 50
    num_candidates = 24
    gamma = 0.10
    min_budget = 5
    max_budget = 199
    max_function_evals = 2000
    nb = NasBench201(file=args.benchmark_file, dataset=args.dataset)
    search_space = NASSearchSpace()
    #experiment = get_nasbench201(name='MOBOHB_NAS_' + str(args.seed), nb=nb, search_space=search_space)
    experiment = get_nasbench201_err_time(name='MOBOHB_NAS_TIMEERR_' + str(args.seed), nb=nb, search_space=search_space)

    ################
    #### MOBOHB ####
    ################
    #
    signal.signal(signal.SIGALRM, signal_handler)  # register the handler
    signal.alarm(args.timeout)
    try:
        get_MOBOHB(
            experiment,
            search_space,
            num_initial_samples=N_init,
            num_candidates=num_candidates,
            gamma=gamma,
            num_iterations=max_function_evals,
            min_budget=min_budget,
            max_budget=max_budget,
            bench=nb
        )
        save_experiment(experiment, f'{experiment.name}.pickle')
    except OutOfTimeException:
        logger.info("catching out of time error and checkpointing the result")

