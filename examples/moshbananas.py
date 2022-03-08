from baselines.problems import get_flowers
from baselines.problems.flowers import FlowersSearchSpace
from baselines.problems import get_fashion
from baselines.problems.fashion import FashionSearchSpace
from baselines import save_experiment
from baselines.methods.mobananas import get_MOSHBANANAS
import signal
import argparse
import time
import os
import sys
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
    parser.add_argument('--timeout', default=85000, type=int, help='Timeout in sec. 0 -> no timeout')
    args = parser.parse_args()
    # Parameters Flowers
    N_init = 10
    min_budget = 5
    max_budget = 25
    max_function_evals = 10000
    num_arch=20
    select_models=10
    eta=3
    search_space = FlowersSearchSpace()
    experiment = get_flowers('MOSHBANANAS_FLOWER'+str(args.seed))

    # Parameters Fashion
    #N_init = 10
    #min_budget = 5
    #max_budget = 25
    #max_function_evals = 1000
    #num_arch=20
    #select_models=10
    #eta=3
    #search_space = FashionSearchSpace()
    #experiment = get_fashion('MOSHBANANAS_'+str(args.seed)+'_'+str(time.time()))
    signal.signal(signal.SIGALRM, signal_handler)  
    signal.alarm(args.timeout)
    #####################
    #### MOSHBANANAS ####
    #####################
    try:
        get_MOSHBANANAS(
            experiment,
            search_space,
            initial_samples=N_init,
            select_models=select_models,
            num_arch=num_arch,
            min_budget=min_budget,
            max_budget=max_budget,
            function_evaluations=max_function_evals,
            eta=eta,
            max_time=args.timeout
        )
        save_experiment(experiment, f'{experiment.name}.pickle')
    except OutOfTimeException:
        logger.info("catching out of time error and checkpointing the result")
        #save_experiment(experiment, f'{experiment.name}.pickle')
