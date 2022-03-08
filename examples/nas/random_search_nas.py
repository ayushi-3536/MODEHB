from ax import Models
import signal

import traceback
import argparse
import sys
from datetime import datetime
from baselines.problems.nasbench201.nas_bench_201 import get_nasbench201_err_time
from baselines.problems.nasbench201.nas_bench_201 import NasBench201
from baselines.problems.nasbench201.nas_search_space import NASSearchSpace
from baselines import save_experiment
from loguru import logger
logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}

parser = argparse.ArgumentParser()
parser.add_argument('--runtime', type=int, default=24*3600*10,
                    help='total runtime')
parser.add_argument('--timeout', default=85000, type=int, help='Timeout in sec. 0 -> no timeout')
parser.add_argument('--seed', type=int, default=500, metavar='S',
                    help='random seed (default: 123)')
parser.add_argument('--benchmark_file',
                     default='/work/dlclarge1/sharmaa-mulobjtest/NATS-tss-v1_0-3ffb9-simple', type=str,
                     help='Timeout in sec. 0 -> no timeout')
#parser.add_argument('--benchmark_file',
#                    default='C:\\users\\ayush\\OneDrive\\Pictures\\nasbench\\NATS-tss-v1_0-3ffb9-simple', type=str,
#                    help='Timeout in sec. 0 -> no timeout')
parser.add_argument('--dataset', default='ImageNet16-120', type=str, help='Timeout in sec. 0 -> no timeout')

args = parser.parse_args()

class OutOfTimeException(Exception):
    # Custom exception for easy handling of timeout
    pass

# Parameters NasBench201
N_init = 50
max_function_evals = 1000

nb = NasBench201(file=args.benchmark_file, dataset=args.dataset)
search_space = NASSearchSpace()
#experiment = get_nasbench201(name='RS_NAS_' + str(args.seed), nb=nb, search_space=search_space)
experiment = get_nasbench201_err_time(name='RS_NAS_TIME_ERR_' + str(args.seed), nb=nb, search_space=search_space)



def signal_handler(sig, frame):
    save_experiment(experiment, f'{experiment.name}.pickle')
    logger.info('Job is being cancelled')
    raise OutOfTimeException


if __name__ == '__main__':

    # signal.signal(signal.SIGALRM, signal_handler)  # register the handler
    # signal.alarm(args.timeout)
    ###############################
    #### Random Search for NAS ####
    ###############################


    curr_time = initial_time = nb.init_time
    try:
       while curr_time - initial_time < int(args.runtime) :
          trial = experiment.new_trial(Models.SOBOL(experiment.search_space).gen(1))
          d = experiment.fetch_data()
          data = d.df
          print("metrices:",d.df['metric_name'].unique())
          tdata = data[(data['metric_name'] == 'train_all_time')]
          traincost = tdata[(tdata['trial_index'] == trial.index)]['mean'].values[0]
          vdata = data[(data['metric_name'] == 'val_per_time')]
          valcost = vdata[(vdata['trial_index'] == trial.index)]['mean'].values[0]

          print("runtime_cost:{},valcost:{}", traincost, valcost)
          runtime_cost = traincost + valcost
          # Artificially add the time
          trial._time_created = datetime.fromtimestamp(curr_time)
          curr_time = curr_time + runtime_cost
          trial._time_completed = datetime.fromtimestamp(curr_time)

          print('Time left: ', int(args.runtime) - (curr_time - initial_time), file=sys.stderr, flush=True)
       logger.info("saving the results")
       save_experiment(experiment, f'{experiment.name}.pickle')
    except (Exception, KeyboardInterrupt, OutOfTimeException) as err:
       traceback.print_exc()
       logger.error("catching error and checkpointing the result:{}",err,exc_info=True)



