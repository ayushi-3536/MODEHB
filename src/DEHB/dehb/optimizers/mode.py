from typing import List

import numpy as np
from loguru import logger
import sys
from .de import DE

logger.configure(handlers=[{"sink": sys.stdout, "level": "ERROR"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}

class MODE(DE):
    def __init__(self, cs=None, f=None, dimensions=None, pop_size=20, max_age=np.inf,
                 mutation_factor=None, crossover_prob=None, strategy='rand1_bin',
                 budget=None, encoding=False, dim_map=None, constraint_model_size=None, constraint_min_precision=None,
                 **kwargs):
        super().__init__(cs=cs, f=f, dimensions=dimensions, pop_size=pop_size, max_age=max_age,
                         mutation_factor=mutation_factor, crossover_prob=crossover_prob,
                         strategy=strategy, budget=budget, **kwargs)
        self.constraint_model_size = constraint_model_size
        self.constraint_min_precision = constraint_min_precision


class MoAsyncDE(MODE):
    def __init__(self, seed=1, cs=None, f=None, dimensions=None, pop_size=None, max_age=np.inf,
                 mutation_factor=None, crossover_prob=None, strategy='rand1_bin',
                 budget=None, async_strategy='deferred', output_path=None, **kwargs):
        '''Extends MODE to be Asynchronous with variations'''
        super().__init__(cs=cs, f=f, dimensions=dimensions, pop_size=pop_size, max_age=max_age,
                         mutation_factor=mutation_factor, crossover_prob=crossover_prob,
                         strategy=strategy, budget=budget, **kwargs)
        if self.strategy is not None:
            self.mutation_strategy = self.strategy.split('_')[0]
            self.crossover_strategy = self.strategy.split('_')[1]
        else:
            self.mutation_strategy = self.crossover_strategy = None
        self.async_strategy = async_strategy
        self.seed = seed

        self.output_path = output_path

        assert self.async_strategy in ['immediate', 'random', 'worst', 'deferred'], \
            "{} is not a valid choice for type of DE".format(self.async_strategy)

    def f_objective(self, x, budget=None, **kwargs):
        if self.f is None:
            raise NotImplementedError("An objective function needs to be passed.")
        if self.encoding:
            x = self.map_to_original(x)
        if self.configspace:
            # converts [0, 1] vector to a ConfigSpace object
            config = self.vector_to_configspace(x)
        else:
            # can insert custom scaling/transform function here
            config = x.copy()
        if budget is not None:  # to be used when called by multi-fidelity based optimizers
            res = self.f(config, self.seed, budget=budget, **kwargs)
        else:
            res = self.f(config, **kwargs)
        assert "fitness" in res
        assert "cost" in res
        return res

    # def init_population(self, pop_size: int, warm_start=False, constraints=None) -> List:
    #
    #     np.random.seed(self.seed)
    #     self.cs.seed(self.seed)
    #     if self.configspace:
    #         # sample from ConfigSpace s.t. conditional constraints (if any) are maintained
    #             configs, costs = bohb_optimization(self.cs, self.seed, pop_size, self.output_path, constraints)
    #             population = configs[:pop_size]
    #             self.fallback_population = configs[-(len(configs)-pop_size):]
    #             logger.debug("population size:{}, pop size:{}", len(population), pop_size)
    #             logger.debug("fallback pop :{}", len(self.fallback_population))
    #
    #             # logger.debug("population:{}",population)
    #             if not isinstance(population, List):
    #                 population = [population]
    #         # the population is maintained in a list-of-vector form where each ConfigSpace
    #         # configuration is scaled to a unit hypercube, i.e., all dimensions scaled to [0,1]
    #             population = [self.configspace_to_vector(individual) for individual in population]
    #     else:
    #         # if no ConfigSpace representation available, uniformly sample from [0, 1]
    #         population = np.random.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))
    #     return np.array(population)


    def _set_min_pop_size(self):
        if self.mutation_strategy in ['rand1', 'rand2dir', 'randtobest1']:
            self._min_pop_size = 3
        elif self.mutation_strategy in ['currenttobest1', 'best1']:
            self._min_pop_size = 2
        elif self.mutation_strategy in ['best2']:
            self._min_pop_size = 4
        elif self.mutation_strategy in ['rand2']:
            self._min_pop_size = 5
        else:
            self._min_pop_size = 1
        self.highest_budget = 5
        return self._min_pop_size

    def mutation(self, seed, current=None, alt_pop=None):
        '''Performs DE mutation
        '''
        if self.mutation_strategy == 'rand1':
            r1, r2, r3 = self.sample_population(size=3, seed=seed, alt_pop=alt_pop)
            mutant = self.mutation_rand1(r1, r2, r3)

        elif self.mutation_strategy == 'rand2':
            r1, r2, r3, r4, r5 = self.sample_population(size=5, seed=seed, alt_pop=alt_pop)
            mutant = self.mutation_rand2(r1, r2, r3, r4, r5)

        elif self.mutation_strategy == 'rand2dir':
            r1, r2, r3 = self.sample_population(size=3, seed=seed, alt_pop=alt_pop)
            mutant = self.mutation_rand2dir(r1, r2, r3)

        return mutant

    def _init_mutant_population(self, pop_size, seed, population=None, target=None, best=None):
        '''Generates pop_size mutants from the passed population
        '''
        population = pop_size
        np.random.seed(seed)
        mutants = np.random.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))
        for i in range(pop_size):
            mutants[i] = self.mutation(current=target, seed=seed, alt_pop=population)
        return mutants

    def sample_population(self, size: int = 3, seed=1, alt_pop: List = None) -> List:
        '''Samples 'size' individuals

        If alt_pop is None or a list/array of None, sample from own population
        Else sample from the specified alternate population (alt_pop)
        '''
        np.random.seed(seed)
        if isinstance(alt_pop, list) or isinstance(alt_pop, np.ndarray):
            idx = [indv is None for indv in alt_pop]
            if any(idx):
                selection = np.random.choice(np.arange(len(self.population)), size, replace=False)
                return self.population[selection]
            else:
                if len(alt_pop) < 3:
                    alt_pop = np.vstack((alt_pop, self.population))
                selection = np.random.choice(np.arange(len(alt_pop)), size, replace=False)
                alt_pop = np.stack(alt_pop)
                return alt_pop[selection]
        else:
            selection = np.random.choice(np.arange(len(self.population)), size, replace=False)
            return self.population[selection]
