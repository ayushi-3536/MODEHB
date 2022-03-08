from copy import deepcopy

from ax import Data, GeneratorRun, Arm
import pandas as pd

from hpbandster.core.worker import Worker
import ConfigSpace as CS
import numpy as np
from datetime import datetime
import sys

class MOBOHBWorker(Worker):
    def __init__(self, experiment, search_space,bench, eval_function, seed=42, **kwargs):
        super().__init__(**kwargs)

        self.experiment = experiment
        self.eval_function = eval_function
        self.search_space = search_space
        self.seed = seed
        self.nb=bench
        self.num_evals=0

    def tchebycheff_norm(self, cost, rho=0.05):
        w = np.random.random_sample(2)
        w /= np.sum(w)

        w_f = w * cost
        max_k = np.max(w_f)
        rho_sum_wf = rho * np.sum(w_f)
        return max_k + rho_sum_wf

    def compute(self, config_id:int, config: CS.Configuration, budget:float, working_directory:str, *args, **kwargs) -> dict:


        params = deepcopy(config)
        params['budget'] = int(budget)

        params['id'] = str(config_id)

        trial = self.experiment.new_trial(GeneratorRun([Arm(params, name=str(config_id))]))
        eval_data = self.experiment.eval_trial(trial)
        #self.num_evals += 1

        data = self.experiment.fetch_data().df
        tdata = data[(data['metric_name'] == 'train_all_time')]
        traincost = tdata[(tdata['trial_index'] == trial.index)]['mean'].values[0]
        vdata = data[(data['metric_name'] == 'val_per_time')]
        valcost = vdata[(vdata['trial_index'] == trial.index)]['mean'].values[0]

        print("runtime_cost:{},valcost:{}", traincost, valcost)
        runtime_cost = traincost + valcost

        # Artificially add the time

        initial_time = self.nb.init_time
        trial._time_created = datetime.fromtimestamp(self.nb.last_ts)
        self.nb.last_ts = self.nb.last_ts + runtime_cost
        trial._time_completed = datetime.fromtimestamp(self.nb.last_ts)
        print("time completed:", trial._time_completed)

        print('Time left: ', 24*3600*10 - (self.nb.last_ts - initial_time), file=sys.stderr, flush=True)

        error = float(eval_data.df[eval_data.df['metric_name'] == 'error']['mean'])
        prediction_time = float(eval_data.df[eval_data.df['metric_name'] == 'norm_prediction_time']['mean'])

        return {'loss': (error, prediction_time)}
