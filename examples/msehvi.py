from baselines.problems.flowers import discrete_flowers
from baselines.problems import get_flowers
from baselines.problems.fashion import discrete_fashion
from baselines.problems import get_fashion
from baselines.problems import get_branin_currin, BraninCurrinEvalFunction
from baselines import save_experiment
from baselines.methods.msehvi.msehvi import MSEHVI
from ax import Models
import signal
import time
import argparse
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

def timeouthandler(signum, frame):
    save_experiment(experiment, f'{experiment.name}.pickle')
    raise OutOfTimeException
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load', default=None, help='checkpoint file to load')
    parser.add_argument('--timeout', default=85000, type=int, help='Timeout in sec. 0 -> no timeout')
    args = parser.parse_args()
    # Parameters Flowers
    N_init = 50 # Number of initial random samples
    N = 20000   # Number of MS-EHVI samples (it is not important)
    discrete_f = discrete_flowers       # Discrete function
    discrete_m = 'num_params'           # Name of the discrete metric
    experiment = get_flowers('MSEHVI_FLOWER'+str(args.seed))  # Function to get the problem

    # Parameters Fashion
    #N_init = 10 # Number of initial random samples
    #N = 20000   # Number of MS-EHVI samples (it is not important)
    #discrete_f = discrete_fashion       # Discrete function
    #discrete_m = 'num_params'           # Name of the discrete metric
    #experiment = get_fashion('MSEHVI_'+str(args.seed)+'_'+str(time.time()))  # Function to get the problem

    # Parameters Branin Crunin
    # N_init = 10
    # N = 100
    # discrete_f = BraninCurrinEvalFunction().discrete_call
    # discrete_m = 'a'
    # experiment = get_branin_currin('MSEHVI')

    #################
    #### MS-EHVI ####
    #################
    signal.signal(signal.SIGALRM, timeouthandler)  # register the handler
    signal.alarm(args.timeout)
    try:
    # Random search initialization

        for _ in range(N_init):
            experiment.new_trial(Models.SOBOL(experiment.search_space).gen(1))
            experiment.fetch_data()
        try:

            msehvi = MSEHVI(experiment, discrete_m, discrete_f)
            for _ in range(N):
                msehvi.step()
        except(KeyboardInterrupt, Exception):
            raise

        save_experiment(experiment, f'{experiment.name}.pickle')
    except OutOfTimeException:
        logger.info("catching out of time error and checkpointing the result")
        print("catching time out of exception")
