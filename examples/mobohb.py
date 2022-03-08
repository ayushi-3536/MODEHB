from baselines.methods.mobohb.run_mobohb import get_MOBOHB
from baselines.problems import get_flowers
from baselines.problems.flowers import FlowersSearchSpace
from baselines.problems import get_fashion
from baselines.problems.fashion import FashionSearchSpace

from baselines.problems import get_wikitext_ppl_score
from baselines.problems.wikitext import WikiSearchSpace
from baselines.problems import get_acc_dsp
from baselines.problems.adult import AdultSearchSpace
from baselines import save_experiment
import sys
import time
import signal
import argparse
from loguru import logger
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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load', default=None, help='checkpoint file to load')
    parser.add_argument('--timeout', default=43200, type=int, help='Timeout in sec. 0 -> no timeout')
    args = parser.parse_args()

    # Parameters Flowers
    #N_init = 50
    #num_candidates = 24
    #gamma = 0.10
    #min_budget = 5
    #max_budget = 25
    #max_function_evals = 2000
    #search_space = FlowersSearchSpace()
    #experiment = get_flowers('MOBOHB_'+str(args.seed))

    # Parameters Fashion
    #N_init = 10
    #num_candidates = 24
    #gamma = 0.10
    #min_budget = 5
    #max_budget = 25
    #max_function_evals = 1500
    #search_space = FashionSearchSpace()
    #experiment = get_fashion('MOBOHB_'+str(args.seed)+'_'+str(time.time()))

    # # Parameters Adult
    # N_init = 10
    # num_candidates = 24
    # gamma = 0.10
    # min_budget = 5
    # max_budget = 200
    # max_function_evals = 1500
    # search_space = AdultSearchSpace()
    # experiment = get_acc_dsp('MOBOHB_Adult_{}'.format(args.seed))

    # Parameters wiki
    N_init = 10
    num_candidates = 24
    gamma = 0.10
    min_budget = 5
    max_budget = 81
    max_function_evals = 3000
    search_space = WikiSearchSpace()
    experiment = get_wikitext_ppl_score('MOBOHB_Wiki_{}'.format(args.seed))

    ################
    #### MOBOHB ####
    ################
    
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
        )
        save_experiment(experiment, f'{experiment.name}.pickle')
    except OutOfTimeException:
        logger.info("catching out of time error and checkpointing the result")

