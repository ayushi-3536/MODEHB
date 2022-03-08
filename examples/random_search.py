from ax import Models
import signal
import argparse
import sys
from datetime import datetime
from baselines.problems import get_flowers
from baselines.problems import get_branin_currin
from baselines.problems import get_acc_dsp
from baselines.problems import get_wikitext_ppl_score
from baselines.problems import get_fashion
from baselines import save_experiment
from loguru import logger
import traceback
import time
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
    parser.add_argument('--timeout', default=86400, type=int, help='Timeout in sec. 0 -> no timeout')
    parser.add_argument('--runtime', default=43200, type=int, help='Timeout in sec. 0 -> no timeout')
    args = parser.parse_args()

    # Parameters Flowers
    # N = 20000   # Number of samples (it is not important)
    # experiment = get_flowers('Flower_RandomSearch'+'_'+str(args.seed))  # Function to get the problem

    # Parameters Fashion
    #N = 20000   # Number of samples (it is not important)
    #experiment = get_fashion('RandomSearch' + '_' + str(args.seed))  # Function to get the prob

    # Parameters Adult
    # N = 400  # Number of samples (it is not important)
    # experiment = get_acc_dsp('RandomSearch_adult' + '_' + str(args.seed))  # Function to get the prob

    #Parameters Wikitext
    N = 1000  # Number of samples (it is not important)
    experiment = get_wikitext_ppl_score('RandomSearch_wikitxt' + '_' + str(args.seed))

    # signal.signal(signal.SIGALRM, signal_handler)  # register the handler
    # signal.alarm(args.timeout)

    ###############################
    #### Random Search for args.runtime ####
    ###############################

    initial_time = time.time()
    try:
       while time.time() - initial_time < args.runtime:
          experiment.new_trial(Models.SOBOL(experiment.search_space).gen(1))
          experiment.fetch_data()
          logger.debug('Time left: {}', args.runtime - (time.time() - initial_time), file=sys.stderr, flush=True)

       save_experiment(experiment, f'{experiment.name}.pickle')
    except (Exception, KeyboardInterrupt, OutOfTimeException) as err:
       traceback.print_exc()
       logger.info("catching error and checkpointing the result:{}",err)




    #######################
    #### Random Search for N evals ####
    #######################

    #
    # try:
    #      for _ in range(N):
    #          experiment.new_trial(Models.SOBOL(experiment.search_space).gen(1))
    #          experiment.fetch_data()
    #
    #      save_experiment(experiment, f'{experiment.name}.pickle')
    # except (Exception, KeyboardInterrupt, OutOfTimeException):
    #      logger.info("catching out of time error and checkpointing the result")
    #      print("catching time out of exception")
