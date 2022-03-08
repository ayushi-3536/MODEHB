import enum
import uuid
from copy import deepcopy

import numpy as np
from typing import Optional, Dict

from ax import Experiment, GeneratorRun, Arm
from loguru import logger
import sys
from datetime import datetime

logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}

class Recombination(enum.IntEnum):
    NONE = -1  # can be used when only mutation is required
    UNIFORM = 0  # uniform crossover (only really makes sense for function dimension > 1)
    INTERMEDIATE = 1  # intermediate recombination

class Mutation(enum.IntEnum):
    NONE = -1  # Can be used when only recombination is required
    UNIFORM = 0  # Uniform mutation
    GAUSSIAN = 1  # Gaussian mutation

class ParentSelection(enum.IntEnum):
    NEUTRAL = 0
    FITNESS = 1
    TOURNAMENT = 2


class Member:
    """
    Class to simplify member handling.
    """

    def __init__(self, search_space,
                 budget: int,
                 mutation: Mutation,
                 recombination: Recombination,
                 name_file: Optional[str] = 'eash',
                 sigma: Optional[float] = None,
                 recom_prob: Optional[float] = None,
                 x_coordinate: Optional[Dict] = None,
                 experiment: Experiment = None,
                 seed = None,
                 bench = None) -> None:
        """
        Init
        :param initial_x: Initial coordinate of the member
        :param target_function: The target function that determines the fitness value
        :param mutation: hyperparameter that determines which mutation type use
        :param recombination: hyperparameter that determines which recombination type to use
        :param sigma: Optional hyperparameter that is only active if mutation is gaussian
        :param recom_prob: Optional hyperparameter that is only active if recombination is uniform
        """
        self._space = search_space
        self._id = uuid.uuid4()
        self._name_file = name_file
        if(seed):
            search_space.seed(seed)
            self._x = search_space.sample_configuration().get_dictionary() if not x_coordinate else x_coordinate
            logger.debug("configuration:{}",self._x)
        else:
            self._x = search_space.sample_configuration().get_dictionary() if not x_coordinate else x_coordinate
        self._age = 0  # basically indicates how many offspring were generated from this member
        self._mutation = mutation
        self._recombination = recombination
        self._x_changed = True
        self._fit = None
        self._sigma = sigma
        self._recom_prob = recom_prob
        self._budget = budget
        self._experiment = experiment
        self._num_evals = 0
        self.nb=bench
        # self.logger = logging.getLogger(self.__class__.__name__)

    @property  # fitness can only be queried never set
    def fitness(self):
        if self._x_changed:  # Only if the x_coordinate or the budget has changed we need to evaluate the fitness.
            self._x_changed = False

            ############ AX THINGS ##############
            params = deepcopy(self._x)
            params['budget'] = int(self._budget)

            trial_name = '{}-{}'.format(self._id, self._num_evals)
            params['id'] = trial_name


            trial = self._experiment.new_trial(GeneratorRun([Arm(params, name=trial_name)]))
            data_eval = self._experiment.eval_trial(trial)
            self._num_evals += 1

            data = self._experiment.fetch_data().df
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


            # acc = float(data_eval.df[data_eval.df['metric_name'] == 'val_acc']['mean'])
            # len = float(data_eval.df[data_eval.df['metric_name'] == 'num_params']['mean'])
            error = float(data_eval.df[data_eval.df['metric_name'] == 'error']['mean'])
            prediction_time = float(data_eval.df[data_eval.df['metric_name'] == 'norm_prediction_time']['mean'])

            self._fit =[error, prediction_time]

        return self._fit  # otherwise we can return the cached value

    @property  # properties let us easily handle getting and setting without exposing our private variables
    def x_coordinate(self):
        return self._x

    @x_coordinate.setter
    def x_coordinate(self, value):
        self._x_changed = True
        self._x = value

    @property
    def budget(self):
        return self._budget

    @budget.setter
    def budget(self, value):
        self._x_changed = True
        self._budget = value

    @property
    def id(self):
        return self._id

    def mutate(self):
        """
        Mutation which creates a new offspring
        :return: new member who is based on this member
        """
        new_x = self.x_coordinate.copy()

        # self.logger.debug('new point before mutation:')
        # self.logger.debug(new_x)

        if self._mutation == Mutation.UNIFORM:
            keys = np.random.choice(list(new_x.keys()), 5, replace=False)
            for k in keys:
                k = str(k)
                if self._space.is_mutable_hyperparameter(k):
                    new_x[k] = self._space.sample_hyperparameter(k)

        elif self._mutation != Mutation.NONE:
            # We won't consider any other mutation types
            raise NotImplementedError

        child = Member(self._space, self._budget, self._mutation, self._recombination, self._name_file,
                       self._sigma, self._recom_prob, new_x, self._experiment,bench=self.nb)
        self._age += 1
        return child

    def recombine(self, partner):
        """
        Recombination of this member with a partner
        :param partner: Member
        :return: new offspring based on this member and partner
        """
        #if self._recombination == Recombination.INTERMEDIATE:
        #    new_x = 0.5 * (self.x_coordinate + partner.x_coordinate)
        if self._recombination == Recombination.UNIFORM:
            assert self._recom_prob is not None, \
                'for this recombination type you have to specify the recombination probability'

            new_x = self.x_coordinate.copy()
            for k in new_x.keys():
                if (np.random.rand() >= self._recom_prob) and (k in partner.x_coordinate.keys()) \
                   and (self._space.is_mutable_hyperparameter(k)):
                    new_x[k] = partner.x_coordinate[k]

        elif self._recombination == Recombination.NONE:
            new_x = self.x_coordinate.copy()  # copy is important here to not only get a reference
        else:
            raise NotImplementedError
        
        # self.logger.debug('new point after recombination:')
        # self.logger.debug(new_x)
        
        child = Member(self._space, self._budget, self._mutation, self._recombination, self._name_file,
                       self._sigma, self._recom_prob, new_x, self._experiment,bench=self.nb)
        self._age += 1
        return child

    def __str__(self):
        """Makes the class easily printable"""
        str = "Population member: Age={}, budget={}, x={}, f(x)={}".format(self._age, self._budget, self.x_coordinate, self.fitness)
        return str

    def __repr__(self):
        """Will also make it printable if it is an entry in a list"""
        return self.__str__() + '\n'
