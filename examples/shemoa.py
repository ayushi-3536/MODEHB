from baselines.problems import get_flowers
from baselines.problems.flowers import FlowersSearchSpace
from baselines.problems import get_fashion
from baselines.problems.fashion import FashionSearchSpace
from baselines.problems import get_wikitext_ppl_score
# from baselines.problems.wikitext import WikiSearchSpace
# from baselines.problems import get_acc_dsp
from baselines.problems.adult import AdultSearchSpace
from baselines import save_experiment
from baselines.methods.shemoa import SHEMOA
from baselines.methods.shemoa import Mutation, Recombination, ParentSelection
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

def timeouthandler(signum, frame):
    print('The end is nigh')
    raise OutOfTimeException

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--load', default=None, help='checkpoint file to load')
    parser.add_argument('--timeout', default=86000, type=int, help='Timeout in sec. 0 -> no timeout')
    args = parser.parse_args()

    # Parameters Flowers
    N_init = 100
    min_budget = 5
    max_budget = 25
    max_function_evals = 15000
    mutation_type = Mutation.UNIFORM
    recombination_type = Recombination.UNIFORM
    selection_type = ParentSelection.TOURNAMENT
    search_space = FlowersSearchSpace()
    experiment = get_flowers('SHEMOA_Flower_'+str(args.seed))

    # # Parameters Adult
    # N_init = 100
    # min_budget = 5
    # max_budget = 200
    # max_function_evals = 15000
    # mutation_type = Mutation.UNIFORM
    # recombination_type = Recombination.UNIFORM
    # selection_type = ParentSelection.TOURNAMENT
    # search_space = AdultSearchSpace()
    # experiment = get_acc_dsp('SHEMOA_Adult_{}'.format(args.seed))

    # Parameters Wiki
    #N_init = 20
    #min_budget = 5
    #max_budget = 81
    #max_function_evals = 170
    #mutation_type = Mutation.UNIFORM
    #recombination_type = Recombination.UNIFORM
    #selection_type = ParentSelection.TOURNAMENT
    #search_space = WikiSearchSpace()
    #experiment = get_wikitext_ppl_score('SHEMOA_wiki_{}'.format(args.seed))

    # Parameters Fashion
    #N_init = 10
    #min_budget = 5
    #max_budget = 25
    #max_function_evals = 1500
    #mutation_type = Mutation.UNIFORM
    #recombination_type = Recombination.UNIFORM
    #selection_type = ParentSelection.TOURNAMENT
    #search_space = FashionSearchSpace()
    #experiment = get_fashion('SHEMOA_15k'+ str(args.seed)+str(time.time()))

    #################
    #### SH-EMOA ####
    #################
    signal.signal(signal.SIGALRM, timeouthandler)  # register the handler
    signal.alarm(args.timeout)
    try:
        ea = SHEMOA(
            search_space,
            experiment,
            N_init, min_budget, max_budget,
            mutation_type=mutation_type,
            recombination_type=recombination_type,
            selection_type=selection_type,
            total_number_of_function_evaluations=max_function_evals
        )
        ea.optimize()
        save_experiment(experiment, f'{experiment.name}.pickle')
    except OutOfTimeException:
        print("catching time out of exception")
        save_experiment(experiment, f'{experiment.name}.pickle')
