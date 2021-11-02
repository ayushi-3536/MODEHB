from .problems.flowers import FlowersSearchSpace
from ..DEHB.dehb.optimizers import MODEHB
import sys
from loguru import logger
import time
from .default_utils import *
import json
from .problems.flowers import evaluate_network

logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}

def input_arguments():
    parser = default_arguments()
    #add experiment specific arguments here
    args = parser.parse_args()
    return args


def objective_function(cfg, seed, budget, run=1, **kwargs):
    start = time.time()
    metrics = evaluate_network(cfg,budget=int(budget))
    acc = metrics['val_acc_1']
    cost = time.time()-start
    total_model_params = metrics['num_params']
    logger.info("budget:{}, numparams:{}, acc:{}",budget,total_model_params,acc)
    with open(output_path + 'dehb_run.json', 'a+')as f:
        json.dump({'configuration': dict(cfg), 'acc': acc,
                   'n_params': total_model_params, 'num_epochs': budget,'cost':cost}, f)

        f.write("\n")

    return ({"cost": cost,
             "fitness": [acc,total_model_params]})  # Because minimize!



args = input_arguments()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False
output_path = create_output_dir(args)
output_path = args.output_path + "_" + str(args.run_id) + '/'
os.makedirs(output_path, exist_ok=True)
cs = FlowersSearchSpace()

def rs():
        for i in range(args.n_iters):
            config = cs.sample_configuration()
            run_info = objective_function(config)
            fitness, cost = run_info["fitness"], run_info["cost"]
            with open(output_path + 'all_runs.json', 'a+')as f:
                f.write(str(fitness))
                f.write("\n")




modehb = MODEHB(objective_function=objective_function,
                   cs=FlowersSearchSpace(),
                   dimensions=len(cs.get_hyperparameters()),
                   min_budget=args.min_budget,
                   max_budget=args.max_budget,
                   eta=args.eta,
                   output_path=output_path,
                   # if client is not None and of type Client, n_workers is ignored
                   # if client is None, a Dask client with n_workers is set up
                   n_workers=args.n_workers,
                   seed=args.seed,
                   log_interval = args.log_interval,
                   ref_point=[0, 8])
modehb.run(total_cost=args.runtime)
