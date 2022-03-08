import json
import os
import traceback
import sys
import time
import numpy as np
from loguru import logger
from distributed import Client
from .de import AsyncDE
from ..optimizers import DEHB
from ..utils import multi_obj_util
import random
logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}


class MODEHB(DEHB):
    def __init__(self, output_path,
                 cs=None,
                 objective_function=None,
                 dimensions=None,
                 mutation_factor=0.5,
                 crossover_prob=0.5,
                 mutation_strategy='rand1_bin',
                 mo_strategy = 'NSGA-II',
                 mo_selection_strategy = 'V2',
                 min_budget=None,
                 max_budget=None,
                 eta=3,
                 min_clip=None,
                 max_clip=None,
                 configspace=True,
                 boundary_fix_type='random',
                 max_age=np.inf,
                 n_workers=None,
                 **kwargs):
        super().__init__(cs=cs, f=objective_function, dimensions=dimensions, mutation_factor=mutation_factor,
                         crossover_prob=crossover_prob, strategy=mutation_strategy, min_budget=min_budget,
                         max_budget=max_budget, eta=eta, min_clip=min_clip, max_clip=max_clip,
                         configspace=configspace, boundary_fix_type=boundary_fix_type, n_workers=n_workers,
                         max_age=max_age, **kwargs)
        self.best_pareto_config = []
        self.pareto_pop = []
        self.pareto_fit = []
        self.de_params = {
            "mutation_factor": self.mutation_factor,
            "crossover_prob": self.crossover_prob,
            "strategy": self.strategy,
            "configspace": self.configspace,
            "boundary_fix_type": self.fix_type,
            "max_age": self.max_age,
            "cs": self.cs,
            "dimensions": self.dimensions,
            "f": objective_function
        }
        self.output_path = output_path
        self.mo_strategy = mo_strategy
        self.mo_selection_strategy = mo_selection_strategy
        self.count_eval = 0
        self.ref_point = kwargs['ref_point']
        self.log_interval = 1000
        self.initial_fitness_threshold = 0.001
        self._init_subpop()



    def _init_subpop(self):
        """ List of DE objects corresponding to the budgets (fidelities)
        """
        logger.info("creating MOSyncDE pop")
        self.de = {}
        pop_count = 0
        logger.info("ref point :{}",self.ref_point)
        for i, b in enumerate(self._max_pop_size.keys()):
            logger.debug("max pop size:{}",self._max_pop_size[b])
            self.de[b] = AsyncDE(**self.de_params, budget=b, pop_size=self._max_pop_size[b])
            self.de[b].population = self.de[b].init_population(pop_size=self._max_pop_size[b])

            self.de[b].fitness = np.array([np.array(self.ref_point) - self.initial_fitness_threshold] * self._max_pop_size[b])
            logger.debug("init pop size:{}, pop fit obtained:{}", self._max_pop_size[b], self.de[b].fitness)

            self.de[b].global_parent_id = np.array([pop_count + counter for counter in range(self._max_pop_size[b])])
            logger.debug("global parent id:{}", self.de[b].global_parent_id)

            # adding attributes to DEHB objects to allow communication across subpopulations
            self.de[b].parent_counter = 0
            self.de[b].promotion_pop = None
            self.de[b].promotion_fitness = None
            pop_count += self._max_pop_size[b]

    def _get_paretos(self, population_fitness):
        index_list = np.array(list(range(len(population_fitness))))
        fitness = np.array([[x[0], x[1]] for x in population_fitness])
        return multi_obj_util.nDS_index(np.array(fitness), index_list)

    def _select_best_config_mo_cd(self, population_fitness):
        a, index_return_list, _ = self._get_paretos(population_fitness)

        logger.debug("a:{},index list :{}",a,index_return_list)
        b, sort_index = multi_obj_util.crowdingDist(a, index_return_list)
        logger.debug("b:{},sort_index:{}",b,sort_index)

        sorted = []
        for x in sort_index:
            sorted.extend(x)
        logger.debug("sorted:{}",sorted)
        return sorted

    def _select_best_config_epsnet(self, population_fitness):
        index_list = np.array(list(range(len(population_fitness))))
        return multi_obj_util.get_eps_net_ranking(population_fitness, index_list)



    def _select_best_hv(self, population_fitness):
        fronts, index_return_list = self._get_paretos(population_fitness)

        logger.debug("fronts:{},index list :{}",fronts,index_return_list)
        sorted = []
        for idx,front in enumerate(fronts):
            hv = np.array(multi_obj_util.contributionsHV3D(front, self.ref_point))
            logger.debug("hv:{}",hv)
            sort_index= np.argsort(-1*hv)
            indexes= index_return_list[idx]
            logger.debug("sort index:{},inrl:{}", sort_index, indexes)
            sorted.extend(indexes[sort_index])
            logger.debug("sort_index:{}",sorted)

        return sorted


    def _concat_all_budget_pop(self, exclude_budget=None):
        """ Concatenates all subpopulations
        """
        budgets = list(self.budgets)
        if exclude_budget is not None:
            budgets.remove(exclude_budget)
        pop = []
        fit = []
        for budget in budgets:
            pop.extend(self.de[budget].population.tolist())
            fit.extend(self.de[budget].fitness.tolist())
        return np.array(pop),fit

    def _get_info_by_global_parent_id(self,p_id):
            """ Concatenates all subpopulations
            """
            budgets = list(self.budgets)
            logger.debug("finding budget and parent id for global pop id:{}",p_id)
            for budget in budgets:
                for idx,g_pid in enumerate(self.de[budget].global_parent_id):
                    if(g_pid == p_id):
                      logger.debug("found budget:{}, parent id :{}",budget,idx)
                      return budget,idx


    def _update_pareto(self):
        """ Concatenates all subpopulations
        """
        pop, fit = self._concat_all_budget_pop()
        fitness = np.array([[x[0], x[1]] for x in fit])
        index_list = np.array(list(range(len(fit))))
        is_pareto, _ = multi_obj_util.pareto_index(fitness,index_list)
        logger.debug("is pareto :{}",is_pareto)
        logger.debug("pop:{}",pop)
        pop = np.array(pop)
        fit = np.array(fit)
        self.pareto_fit = fit[is_pareto, :]
        logger.debug("fit:{}",self.pareto_fit)
        self.pareto_pop = pop[is_pareto,:]

        logger.debug("pareto pop :{}",self.pareto_pop)
        logger.debug("pareto fit:{}",self.pareto_fit)
        fitness = np.array([[x[0], x[1]] for x in self.pareto_fit])
        contributions = multi_obj_util.contributionsHV3D(fitness, self.ref_point)
        contributions_index = np.argsort(contributions)
        contributions_index = contributions_index[::-1]
        self.best_pareto_config = [self.pareto_pop[int(m)] for m in contributions_index]

        if (self.count_eval % self.log_interval == 0):
            with open(os.path.join(self.output_path, "pareto_fit_{}.txt".format(time.time())), 'w') as f:
                np.savetxt(f, self.pareto_fit)
        if (self.count_eval % self.log_interval == 0):
            with open(os.path.join(self.output_path, "pareto_pop_{}.txt".format(time.time())), 'w') as f:
              for item in self.pareto_pop:
                f.write(str(item))


    def _get_promotion_candidate(self, low_budget, high_budget, n_configs):
        """ Manages the population to be promoted from the lower to the higher budget.

        This is triggered or in action only during the first full HB bracket, which is equivalent
        to the number of brackets <= max_SH_iter.
        """
        lbf = np.array(self.de[low_budget].fitness)
        #evaluated_configs = np.where([(lbf[:, 0] < np.inf) & (lbf[:, 1] < np.inf)])[1]
        evaluated_configs = np.where([(lbf[:, 0] < self.ref_point[0] - 0.01) & (lbf[:, 1] < self.ref_point[1] - 0.01)])[1]
        # np.where(self.de[low_budget].fitness != [np.inf,np.inf,np.inf])
        promotion_candidate_pop = [self.de[low_budget].population[config] for config in evaluated_configs]
        promotion_candidate_fitness = [self.de[low_budget].fitness[config] for config in evaluated_configs]

        # ordering the evaluated individuals based on their fitness values
        if(self.mo_strategy == 'NSGA-II'):
            pop_idx = self._select_best_config_mo_cd(population_fitness=promotion_candidate_fitness)
        elif(self.mo_strategy == 'EPSNET'):
            pop_idx = self._select_best_config_epsnet(population_fitness=promotion_candidate_fitness)
        elif(self.mo_strategy == 'MAXHV'):
            pop_idx = self._select_best_hv(population_fitness=promotion_candidate_fitness)

        # creating population for promotion if none promoted yet or nothing to promote
        if self.de[high_budget].promotion_pop is None or \
                len(self.de[high_budget].promotion_pop) == 0:
            self.de[high_budget].promotion_pop = np.empty((0, self.dimensions))
            self.de[high_budget].promotion_fitness = np.array([])

            # iterating over the evaluated individuals from the lower budget and including them
            # in the promotion population for the higher budget only if it's not in the population
            # this is done to ensure diversity of population and avoid redundant evaluations
            for idx in pop_idx:
                individual = promotion_candidate_pop[idx]
                # checks if the candidate individual already exists in the high budget population
                if np.any(np.all(individual == self.de[high_budget].population, axis=1)):
                    # skipping already present individual to allow diversity and reduce redundancy
                    continue
                self.de[high_budget].promotion_pop = np.append(
                    self.de[high_budget].promotion_pop, [individual], axis=0
                )
                self.de[high_budget].promotion_fitness = np.append(
                    self.de[high_budget].promotion_pop, promotion_candidate_fitness[idx]
                )

            # retaining only n_configs
            self.de[high_budget].promotion_pop = self.de[high_budget].promotion_pop[:n_configs]
            self.de[high_budget].promotion_fitness = \
                self.de[high_budget].promotion_fitness[:n_configs]

        if len(self.de[high_budget].promotion_pop) > 0:
            config = self.de[high_budget].promotion_pop[0]
            # removing selected configuration from population
            self.de[high_budget].promotion_pop = self.de[high_budget].promotion_pop[1:]
            self.de[high_budget].promotion_fitness = self.de[high_budget].promotion_fitness[1:]
        else:
            # in case of an edge failure case where all high budget individuals are same
            # just choose the best performing individual from the lower budget (again)
            config = self.de[low_budget].population[pop_idx[0]]

        return config



    ##############################MODEHB SELECTION V101###################################################################3
    def _acquire_config(self, bracket, budget):
        """ Generates/chooses a configuration based on the budget and iteration number
        """
        # select a parent/target
        parent_id = self._get_next_parent_for_subpop(budget)
        target = self.de[budget].population[parent_id]
        configs = [self.vector_to_configspace(config) for config in self.de[budget].population]

        # identify lower budget/fidelity to transfer information from
        lower_budget, num_configs = bracket.get_lower_budget_promotions(budget)
        logger.debug("targetparent:{}, budget:{},lower budget:{},num configs:{}",parent_id,budget,lower_budget,num_configs)

        # logger.debug("num config to be obtained {},loerr_budget:{}", num_configs,lower_budget)
        configs = [self.vector_to_configspace(config) for config in self.de[lower_budget].population]
        logger.debug("iteration counter:{}, max sh iteration:{}", self.iteration_counter, self.max_SH_iter)
        if self.iteration_counter < self.max_SH_iter:
            # promotions occur only in the first set of SH brackets under Hyperband
            # for the first rung/budget in the current bracket, no promotion is possible and
            # evolution can begin straight away
            # for the subsequent rungs, individuals will be promoted from the lower_budget
            if budget != bracket.budgets[0]:

                config = self._get_promotion_candidate(lower_budget, budget, num_configs)
                logger.debug("getting promotion candidate with bracket budget:{}: {}", bracket.budgets[0],
                             self.iteration_counter)
                return config, parent_id
            else:
                logger.debug("not getting promotion candidate,{}, and max {}", budget, bracket.budgets[0])
        # DE evolution occurs when either all individuals in the subpopulation have been evaluated
        # at least once, i.e., has fitness < np.inf, which can happen if
        # iteration_counter <= max_SH_iter but certainly never when iteration_counter > max_SH_iter

        # a single DE evolution --- (mutation + crossover) occurs here

        if(self.mo_strategy == 'NSGA-II'):
            mutation_pop_idx = self._select_best_config_mo_cd(self.de[lower_budget].fitness)[:num_configs]
        elif(self.mo_strategy == 'EPSNET'):
            mutation_pop_idx = self._select_best_config_epsnet(self.de[lower_budget].fitness)[:num_configs]
        elif(self.mo_strategy == 'MAXHV'):
            mutation_pop_idx = self._select_best_hv(self.de[lower_budget].fitness)[:num_configs]


        mutation_pop = self.de[lower_budget].population[mutation_pop_idx]

        # logger.debug("fronts:{}",fronts)
        logger.debug("mutation pop:{}, mutation_pop_idx:{}",mutation_pop,mutation_pop_idx)


        # generate mutants from previous budget subpopulation or global population
        if len(mutation_pop) < self.de[budget]._min_pop_size:
            filler = self.de[budget]._min_pop_size - len(mutation_pop) + 1
            logger.debug("concating all budget pop, mutation_pop:{}", mutation_pop)
            sub_pop,fit = self._concat_all_budget_pop()
            logger.debug("sub pop :{}",sub_pop)
            logger.debug("sub pop fit :{}", fit)
            new_pop = self.de[budget]._init_mutant_population(
                pop_size=filler, population=sub_pop)
            logger.debug("new pop :{}", new_pop)
            mutation_pop = np.concatenate((mutation_pop, new_pop))
            logger.debug("final pop :{}", mutation_pop)



        logger.debug("final pop :{}", mutation_pop)
        mutant = self.de[budget].mutation(
            current=target, alt_pop=mutation_pop
        )
        logger.debug("mutant is :{}, target is :{}", mutant, target)
        # perform crossover with selected parent
        config = self.de[budget].crossover(target=target, mutant=mutant)
        config = self.de[budget].boundary_check(config)
        logger.debug("config after cross over:{}", self.vector_to_configspace(config))
        return config, parent_id






    ################################################### MODEHB V1 #######################################################
    ''' This function checks the fitness of parent and evaluated config and replace parent only if its fitness(evaluated by NDS and hypervolume)
        is greater than the parent, IF BOTH LIES ON THE SAME FRONT, REPLACE THE ONE WITH SMALLER HV CONTRIBUTION
        {IF PARENT AND CHILD BOTH LIE ON THE FIRST FRONT, IT DECREASES HV}'''
    # def check_fitness(self, current_fitness, parent_idx, pop_fitnesses):
    #     fitnesses = pop_fitnesses.copy()
    #     logger.debug("parent_idx:{}",parent_idx)
    #     logger.debug("pop_fitnesses:{}",pop_fitnesses)
    #     logger.debug("curr fit:{}",current_fitness)
    #     fitnesses = np.append(fitnesses, [current_fitness], axis=0)
    #     curr_idx = len(fitnesses) - 1
    #     fitness = np.array([[x[0], x[1]] for x in fitnesses])
    #     index_list = np.array(list(range(len(fitnesses))))
    #     fronts, index_return_list = pareto.nDS_index_front(np.array(fitness), index_list)
    #     for idx, front_index in enumerate(index_return_list):
    #         front_index = front_index
    #         if curr_idx not in front_index and parent_idx not in front_index:
    #             continue
    #         if curr_idx in front_index and parent_idx in front_index:
    #             contributions = pareto.computeHV3D(fronts[idx],self.ref_point)
    #             front_curr_index = np.where(front_index == curr_idx)[0][0]
    #             front_parent_index = np.where(front_index == parent_idx)[0][0]
    #             hv_parent = contributions[front_parent_index]
    #             hv_curr = contributions[front_curr_index]
    #             logger.debug("hv contri parent:{}, hv contri child:{}", hv_parent, hv_curr)
    #             if (hv_parent < hv_curr):
    #                 logger.debug("choosing current:{}", current_fitness)
    #                 return True
    #             logger.debug("choosing parent")
    #             return False
    #
    #         elif curr_idx in front_index and parent_idx not in front_index:
    #             logger.debug("chhose child from front first")
    #             return True
    #         else:
    #             logger.debug("chhose parent from front first")
    #             return False


    ################################    MODEHB V2   ################################################################3
    # ''' This function checks the fitness of parent and evaluated config and replace parent only
    # if its fitness(evaluated by NDS and hypervolume) is greater than the parent, if both lies on the same front,
    # replace least hv contributor'''
    #
    def check_fitness_V2(self, current_fitness, parent_fitness, global_parent_id, parent_id,budget, config):
        pop, fit = self._concat_all_budget_pop()
        pop = pop.tolist()
        target = self.de[budget].population[parent_id]
        logger.debug("all population fitness:{}", fit)
        logger.debug("parent_fitness:{}", parent_fitness)
        logger.debug("curr fitness:{}", current_fitness)
        logger.debug("target:{}",target)
        logger.debug("pop:{}",pop)
        fit.extend([current_fitness])
        pop.extend([config])
        curr_idx = len(fit)-1
        parent_idx = global_parent_id
        logger.debug("parent idx:{}",parent_idx)
        fitness = np.array([[x[0], x[1]] for x in fit])
        index_list = np.array(list(range(len(fit))))
        logger.debug("fitness:{}",fitness)
        logger.debug("index_list:{}",index_list)
        fronts, _, index_return_list = multi_obj_util.nDS_index(np.array(fitness), index_list)

        logger.debug("fronts:{}",fronts)
        logger.debug("fronts index:{}",index_return_list[0])

        for idx, front_index in enumerate(index_return_list):
            front_index = front_index
            if curr_idx not in front_index and parent_idx not in front_index:
                continue
            if curr_idx in front_index and parent_idx in front_index:
                logger.debug("parent and child both in front first, replacing the worst contributor of concat pop with the config")

                idx = multi_obj_util.minHV3D(fitness, self.ref_point)
                if(idx == curr_idx):
                    logger.debug("worst candidate with idx:{} is current index, not replacing any config",idx)
                    return
                budget, parent_id = self._get_info_by_global_parent_id(idx)
                logger.debug("global idx:{},budget:{},parentid:{}",idx,budget,parent_id)
                logger.debug("replacing config from budget:{} and parent_id:{} "
                             "with least hv contribution and fitness:{} with config with fitness:{}",budget,
                             parent_id,self.de[budget].fitness[parent_id],current_fitness)
                self.de[budget].population[parent_id] = config
                self.de[budget].fitness[parent_id] = np.array(current_fitness)
                return

            elif curr_idx in front_index and parent_idx not in front_index:
                ''' Updating the population by replacing parent with child
                '''
                logger.debug("chhose child from front first")
                self.de[budget].population[parent_id] = config
                logger.debug("config in parents place :{}", self.vector_to_configspace(config))
                configs = [self.vector_to_configspace(config) for config in self.de[budget].population]
                logger.debug("modifies budget configs:{}", configs)
                self.de[budget].fitness[parent_id] = np.array(current_fitness)
                return
            else:
                logger.debug("choose parent")
                return



    # ########################  MODEHB - V3 ###################################################################
    # '''replace least hv contributor of all the sub population (not sure how it works, shows good perf though
    # after some time plateus, also deviates from DE concept of selection bwn parent and target)'''
    def check_fitness_V3(self, current_fitness, parent_fitness, global_parent_id, parent_id,budget, config):
        pop, fit = self._concat_all_budget_pop()
        pop = pop.tolist()
        target = self.de[budget].population[parent_id]
        logger.debug("all population fitness:{}", fit)
        logger.debug("parent_fitness:{}", parent_fitness)
        logger.debug("curr fitness:{}", current_fitness)
        logger.debug("target:{}",target)
        logger.debug("pop:{}",pop)
        fit.extend([current_fitness])
        pop.extend([config])
        curr_idx = len(fit)-1
        parent_idx = global_parent_id
        logger.debug("parent idx:{}",parent_idx)
        fitness = np.array([[x[0], x[1]] for x in fit])
        index_list = np.array(list(range(len(fit))))
        logger.debug("fitness:{}",fitness)
        logger.debug("index_list:{}",index_list)
        fronts, _, index_return_list = multi_obj_util.nDS_index(np.array(fitness), index_list)

        logger.debug("fronts:{}",fronts)
        logger.debug("fronts index:{}",index_return_list[0])
        logger.debug("fronts:{}",fronts[-1])
        logger.debug("list:{}",index_return_list)
        last_front_list=index_return_list[-1]
        idx = last_front_list[multi_obj_util.minHV3D(fronts[-1], self.ref_point)]
        if (idx == curr_idx):
            logger.debug("worst candidate with idx:{} is current index, not replacing any config", idx)
            return
        else:

            budget, parent_id = self._get_info_by_global_parent_id(idx)
            logger.debug("global idx:{},budget:{},parentid:{}", idx, budget, parent_id)
            logger.debug("replacing config from budget:{} and parent_id:{} "
                         "with least hv contribution and fitness:{} with config with fitness:{}", budget,
                         parent_id, self.de[budget].fitness[parent_id], current_fitness)
            self.de[budget].population[parent_id] = config
            self.de[budget].fitness[parent_id] = np.array(current_fitness)
            return


    ########################  MODEHB - V4 ###################################################################
    # '''replace min hv contributor of on the same budget'''
    def check_fitness_V4(self, current_fitness, parent_fitness, global_parent_id, parent_id,budget, config):
        pop = []
        fit = []
        pop.extend(self.de[budget].population.tolist())
        fit.extend(self.de[budget].fitness.tolist())

        target = self.de[budget].population[parent_id]
        logger.debug("all population fitness:{}", fit)
        logger.debug("parent_fitness:{}", parent_fitness)
        logger.debug("curr fitness:{}", current_fitness)
        logger.debug("target:{}",target)
        logger.debug("pop:{}",pop)
        fit.extend([current_fitness])
        pop.extend([config])
        curr_idx = len(fit)-1
        parent_idx = parent_id
        logger.debug("parent idx:{}",parent_idx)
        fitness = np.array([[x[0], x[1]] for x in fit])
        index_list = np.array(list(range(len(fit))))
        logger.debug("fitness:{}",fitness)
        logger.debug("index_list:{}",index_list)
        fronts, _, index_return_list = multi_obj_util.nDS_index(np.array(fitness), index_list)

        logger.debug("fronts:{}",fronts)
        logger.debug("fronts index:{}",index_return_list[0])
        logger.debug("fronts:{}",fronts[-1])
        logger.debug("list:{}",index_return_list)
        last_front_list=index_return_list[-1]
        idx = last_front_list[multi_obj_util.minHV3D(fronts[-1], self.ref_point)]
        if (idx == curr_idx):
            logger.debug("worst candidate with idx:{} is current index, not replacing any config", idx)
            return
        else:
            parent_id = idx
            logger.debug("budget:{},parentid:{}", budget, parent_id)
            logger.debug("replacing config from budget:{} and parent_id:{} "
                         "with least hv contribution and fitness:{} with config with fitness:{}", budget,
                         parent_id, self.de[budget].fitness[parent_id], current_fitness)
            self.de[budget].population[parent_id] = config
            self.de[budget].fitness[parent_id] = np.array(current_fitness)
            return



    def _get_next_job(self):
        """ Loads a configuration and budget to be evaluated next by a free worker
        """
        bracket = None
        if len(self.active_brackets) == 0 or \
                np.all([bracket.is_bracket_done() for bracket in self.active_brackets]):
            # start new bracket when no pending jobs from existing brackets or empty bracket list
            bracket = self._start_new_bracket()
        else:
            for _bracket in self.active_brackets:
                # check if _bracket is not waiting for previous rung results of same bracket
                # _bracket is not waiting on the last rung results
                # these 2 checks allow DEHB to have a "synchronous" Successive Halving
                if not _bracket.previous_rung_waits() and _bracket.is_pending():
                    # bracket eligible for job scheduling
                    bracket = _bracket
                    break
            if bracket is None:
                # start new bracket when existing list has all waiting brackets
                bracket = self._start_new_bracket()
        # budget that the SH bracket allots
        budget = bracket.get_next_job_budget()
        config, parent_id = self._acquire_config(bracket, budget)
        global_parent_id = self.de[budget].global_parent_id[parent_id]

        # notifies the Bracket Manager that a single config is to run for the budget chosen
        job_info = {
            "config": config,
            "budget": budget,
            "parent_id": parent_id,
            "global_parent_id": global_parent_id,
            "bracket_id": bracket.bracket_id
        }
        return job_info
    def _save_incumbent(self, fitness, config, name=None):
        self._update_pareto()

    def _update_trackers(self, runtime, history):
        self.runtime.append(runtime)
        self.history.append(history)

    def save_results(self):
        with open(os.path.join(self.output_path, "pareto_fit_{}.txt".format(time.time())), 'w') as f:
                np.savetxt(f, self.pareto_fit)
        with open(os.path.join(self.output_path, "pareto_pop_{}.txt".format(time.time())), 'w') as f:
              for item in self.pareto_pop:
                f.write(str(item))
        if self.history is not None:

                costs = np.array([[x[1][0], x[1][1]] for x in self.history])
                file_name = open(self.output_path + 'every_run_cost_%s.txt' % time.time(), 'w')
                # for line in costs:
                np.savetxt(file_name, costs)
                file_name.close()

    def _fetch_results_from_workers(self):
        """ Iterate over futures and collect results from finished workers
        """
        self.count_eval = self.count_eval + 1
        if self.n_workers > 1 or isinstance(self.client, Client):
            done_list = [(i, future) for i, future in enumerate(self.futures) if future.done()]
        else:
            # Dask not invoked in the synchronous case
            done_list = [(i, future) for i, future in enumerate(self.futures)]
        if len(done_list) > 0:
            self.logger.debug(
                "Collecting {} of the {} job(s) active.".format(len(done_list), len(self.futures))
            )
        for _, future in done_list:
            if self.n_workers > 1 or isinstance(self.client, Client):
                run_info = future.result()
                if "device_id" in run_info:
                    # updating GPU usage
                    self.gpu_usage[run_info["device_id"]] -= 1
                    self.logger.debug("GPU device released: {}".format(run_info["device_id"]))
                future.release()
            else:
                # Dask not invoked in the synchronous case
                run_info = future
            # update bracket information
            logger.debug("run info is {}", run_info)
            fitness, cost = run_info["fitness"], run_info["cost"]
            info = run_info["info"] if "info" in run_info else dict()
            budget, parent_id = run_info["budget"], run_info["parent_id"]
            config = run_info["config"]
            bracket_id = run_info["bracket_id"]
            global_parent_id = run_info["global_parent_id"]
            for bracket in self.active_brackets:
                if bracket.bracket_id == bracket_id:
                    # bracket job complete
                    bracket.complete_job(budget)  # IMPORTANT to perform synchronous SH

            # carry out DE selection
            parent_fitness = self.de[budget].fitness[parent_id]
            logger.debug("budget:{}, parent id:{}",budget, parent_id)
            logger.debug("fitness :{},parent fitness{}", fitness, parent_fitness)
            if(self.mo_selection_strategy == 'V3'):
                self.check_fitness_V3(fitness, parent_fitness,global_parent_id,parent_id,budget,config)
            elif(self.mo_selection_strategy == 'V2'):
                self.check_fitness_V2(fitness, parent_fitness,global_parent_id,parent_id,budget,config)
            elif(self.mo_selection_strategy == 'V4'):
                self.check_fitness_V4(fitness, parent_fitness,global_parent_id,parent_id,budget,config)
            self._update_pareto()
            if self.history is not None and self.count_eval % 10000 == 0:
                logger.debug("history {}", self.history)
                costs = np.array([[x[1][0], x[1][1]] for x in self.history])
                file_name = open(os.path.join(self.output_path + 'every_run_cost_%s.txt' % time.time()), 'w')
                # for line in costs:
                np.savetxt(file_name, costs)
                file_name.close()

            # book-keeping
            if(self.count_eval>1):
                runtime = self.runtime[-1]+cost
            else:
                runtime=cost
            self._update_trackers(
                runtime=runtime,
                history=(config.tolist(), fitness, float(cost), float(budget), info)
            )
            logger.debug("ref point:{}",self.ref_point)
            with open(os.path.join(self.output_path, "hv_contribution_{}.txt"), 'a') as f:
                logger.debug("paeto fit:{}",self.pareto_fit)

                ra = [multi_obj_util.computeHV(self.pareto_fit, self.ref_point), cost, self.count_eval]

                logger.debug("pareto:{}", multi_obj_util.computeHV(self.pareto_fit, self.ref_point))
                np.savetxt(f,ra)

        # remove processed future
        self.futures = np.delete(self.futures, [i for i, _ in done_list]).tolist()
    def _f_objective(self, job_info):
        """ Wrapper to call DE's objective function.
        """
        # check if job_info appended during job submission self.submit_job() includes "gpu_devices"
        if "gpu_devices" in job_info and self.single_node_with_gpus:
            # should set the environment variable for the spawned worker process
            # reprioritising a CUDA device order specific to this worker process
            os.environ.update({"CUDA_VISIBLE_DEVICES": job_info["gpu_devices"]})

        config, budget, parent_id = job_info['config'], job_info['budget'], job_info['parent_id']
        bracket_id = job_info['bracket_id']
        kwargs = job_info["kwargs"]
        res = self.de[budget].f_objective(config, budget, **kwargs)
        info = res["info"] if "info" in res else dict()
        run_info = {
            'fitness': res["fitness"],
            'cost': res["cost"],
            'config': config,
            'budget': budget,
            'parent_id': parent_id,
            'bracket_id': bracket_id,
            'global_parent_id': job_info['global_parent_id'],
            'info': info
        }

        if "gpu_devices" in job_info:
            # important for GPU usage tracking if single_node_with_gpus=True
            device_id = int(job_info["gpu_devices"].strip().split(",")[0])
            run_info.update({"device_id": device_id})
        return run_info
    @logger.catch
    def run(self, total_cost=None, fevals=None, brackets=None, single_node_with_gpus=False,
            verbose=True, debug=False, save_intermediate=True, save_history=True, **kwargs):
        """ Main interface to run optimization by DEHB

        This function waits on workers and if a worker is free, asks for a configuration and a
        budget to evaluate on and submits it to the worker. In each loop, it checks if a job
        is complete, fetches the results, carries the necessary processing of it asynchronously
        to the worker computations.

        The duration of the DEHB run can be controlled by specifying one of 3 parameters. If more
        than one are specified, DEHB selects only one in the priority order (high to low):
        1) Number of function evaluations (fevals)
        2) Number of Successive Halving brackets run under Hyperband (brackets)
        3) Total computational cost (in seconds) aggregated by all function evaluations (total_cost)
        """
        try:
            # checks if a Dask client exists
            if len(kwargs) > 0 and self.n_workers > 1 and isinstance(self.client, Client):
                # broadcasts all additional data passed as **kwargs to all client workers
                # this reduces overload in the client-worker communication by not having to
                # serialize the redundant data used by all workers for every job
                self.shared_data = self.client.scatter(kwargs, broadcast=True)

            # allows each worker to be mapped to a different GPU when running on a single node
            # where all available GPUs are accessible
            self.single_node_with_gpus = single_node_with_gpus
            if self.single_node_with_gpus:
                self.distribute_gpus()

            self.start = time.time()
            if verbose:
                print("\nLogging at {} for optimization starting at {}\n".format(
                    os.path.join(os.getcwd(), self.log_filename),
                    time.strftime("%x %X %Z", time.localtime(self.start))
                ))
            if debug:
                logger.configure(handlers=[{"sink": sys.stdout}])
            while True:
                if self._is_run_budget_exhausted(fevals, brackets, total_cost):
                    break
                if self.is_worker_available():
                    job_info = self._get_next_job()
                    if brackets is not None and job_info['bracket_id'] >= brackets:
                        # ignore submission and only collect results
                        # when brackets are chosen as run budget, an extra bracket is created
                        # since iteration_counter is incremented in _get_next_job() and then checked
                        # in _is_run_budget_exhausted(), therefore, need to skip suggestions
                        # coming from the extra allocated bracket
                        # _is_run_budget_exhausted() will not return True until all the lower brackets
                        # have finished computation and returned its results
                        pass
                    else:
                        if self.n_workers > 1 or isinstance(self.client, Client):
                            self.logger.debug("{}/{} worker(s) available.".format(
                                self._get_worker_count() - len(self.futures), self._get_worker_count()
                            ))
                        # submits job_info to a worker for execution
                        self.submit_job(job_info, **kwargs)
                        if verbose:
                            budget = job_info['budget']
                            self._verbosity_runtime(fevals, brackets, total_cost)
                            self.logger.debug(
                                "Evaluating a configuration with budget {} under "
                                "bracket ID {}".format(budget, job_info['bracket_id'])
                            )
                        self._verbosity_debug()

                self._fetch_results_from_workers()
                if save_history and self.history is not None:
                    self._save_history()
                self.clean_inactive_brackets()
            #logger.debug("constraint tracker is {}", self.history)
            # end of while

            if verbose and len(self.futures) > 0:
                self.logger.debug(
                    "DEHB optimisation over! Waiting to collect results from workers running..."
                )
            while len(self.futures) > 0:
                self._fetch_results_from_workers()
                # if save_intermediate:
                #     self._save_incumbent()
                if save_history and self.history is not None:
                    self._save_history()
                time.sleep(0.05)  # waiting 50ms
            self.save_results()
            return np.array(self.runtime), np.array(self.history, dtype=object), self.pareto_pop, self.pareto_fit

        except(KeyboardInterrupt, Exception) as err:
            logger.error("exception caught:{}",err)
            self.save_results()
            traceback.print_exc()
            return np.array(self.runtime), np.array(self.history, dtype=object), self.pareto_pop, self.pareto_fit
            raise

