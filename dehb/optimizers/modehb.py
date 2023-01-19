import os
import sys
import time
import traceback
from sklearn.preprocessing import normalize
import numpy as np
from distributed import Client
from loguru import logger

from dehb.optimizers import AsyncDE
from dehb.optimizers import DEHB
from dehb.utils import multi_obj_util

logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
_logger_props = {
    "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} {level} {message}",
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
                 mo_strategy='NSGA-II',
                 min_budget=None,
                 max_budget=None,
                 eta=3,
                 min_clip=None,
                 max_clip=None,
                 configspace=True,
                 boundary_fix_type='random',
                 max_age=np.inf,
                 n_workers=None,
                 num_objectives=2,
                 log_interval=100,
                 **kwargs):
        super().__init__(cs=cs, f=objective_function, dimensions=dimensions, mutation_factor=mutation_factor,
                         crossover_prob=crossover_prob, strategy=mutation_strategy, min_budget=min_budget,
                         max_budget=max_budget, eta=eta, min_clip=min_clip, max_clip=max_clip,
                         configspace=configspace, boundary_fix_type=boundary_fix_type, n_workers=n_workers,
                         max_age=max_age, **kwargs)
        self.pareto_pop = []
        self.pareto_fit = []
        self.output_path = output_path
        self.mo_strategy = mo_strategy
        self.count_eval = 0
        self.log_interval = log_interval
        self.num_objective = num_objectives

    def _init_subpop(self):
        """ List of DE objects corresponding to the budgets (fidelities)
        """
        logger.info("creating MOSyncDE pop")
        self.de = {}
        pop_count = 0
        for i, b in enumerate(self._max_pop_size.keys()):
            logger.debug("max pop size:{}", self._max_pop_size[b])
            self.de[b] = AsyncDE(**self.de_params, budget=b, pop_size=self._max_pop_size[b])
            self.de[b].population = self.de[b].init_population(pop_size=self._max_pop_size[b])
            self.de[b].fitness = np.array([np.full(self.num_objective, np.inf).tolist()] * self._max_pop_size[b])
            logger.debug("init pop size:{}, pop fit obtained:{}", self._max_pop_size[b], self.de[b].fitness)
            self.de[b].global_parent_id = np.array([pop_count + counter for counter in range(self._max_pop_size[b])])
            logger.debug("global parent id:{}", self.de[b].global_parent_id)

            # adding attributes to DEHB objects to allow communication across subpopulations
            self.de[b].parent_counter = 0
            self.de[b].promotion_pop = None
            self.de[b].promotion_fitness = None
            pop_count += self._max_pop_size[b]

    def _get_paretos(self, population_fitness):
        index_list = np.array(range(len(population_fitness)))
        fitness = np.array([np.array(x) for x in population_fitness])
        return multi_obj_util.nDS_index(np.array(fitness), index_list)

    def _select_best_config_mo_cd(self, population_fitness):
        a, index_return_list, _ = self._get_paretos(population_fitness)
        b, sort_index = multi_obj_util.crowdingDist(a, index_return_list)
        sorted = []
        for x in sort_index:
            sorted.extend(x)
        return sorted

    def _select_best_config_epsnet(self, population_fitness):
        index_list = np.array(range(len(population_fitness)))
        return multi_obj_util.get_eps_net_ranking(population_fitness, index_list)

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
        return pop, fit

    def _get_info_by_global_parent_id(self, p_id):
        """ Concatenates all subpopulations
        """
        budgets = list(self.budgets)
        for budget in budgets:
            for idx, g_pid in enumerate(self.de[budget].global_parent_id):
                if g_pid == p_id:
                    return budget, idx

    def _update_pareto(self):
        """ Concatenates all subpopulations
        """
        pop, fit = self._concat_all_budget_pop()
        fitness = np.array([np.array(x) for x in fit])
        index_list = np.array(range(len(fit)))
        is_pareto, _ = multi_obj_util.pareto_index(fitness, index_list)
        pop = np.array(pop)
        fit = np.array(fit)
        self.pareto_fit = fit[is_pareto, :]
        self.pareto_pop = pop[is_pareto, :]

    def _get_promotion_candidate(self, low_budget, high_budget, n_configs):
        """ Manages the population to be promoted from the lower to the higher budget.

        This is triggered or in action only during the first full HB bracket, which is equivalent
        to the number of brackets <= max_SH_iter.
        """
        lower_budget_fitness = np.array(self.de[low_budget].fitness)
        evaluated_configs = np.where([(lower_budget_fitness[:, 0] < np.inf) & (lower_budget_fitness[:, 1] < np.inf)])[1]

        # np.where(self.de[low_budget].fitness != [np.inf,np.inf,np.inf])
        promotion_candidate_pop = [self.de[low_budget].population[config] for config in evaluated_configs]
        promotion_candidate_fitness = [self.de[low_budget].fitness[config] for config in evaluated_configs]

        normalized_fitness = np.array(promotion_candidate_fitness).copy()
        normalized_fitness = normalize(normalized_fitness, axis=0, norm='max')

        # ordering the evaluated individuals based on their fitness values
        if self.mo_strategy == 'NSGA-II':
            pop_idx = self._select_best_config_mo_cd(population_fitness=promotion_candidate_fitness)
        else:
            pop_idx = self._select_best_config_epsnet(population_fitness=promotion_candidate_fitness)

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
            self.de[high_budget].promotion_fitness = self.de[high_budget].promotion_fitness[:n_configs]

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

    def _acquire_config(self, bracket, budget):
        """ Generates/chooses a configuration based on the budget and iteration number
        """
        # select a parent/target
        parent_id = self._get_next_parent_for_subpop(budget)
        target = self.de[budget].population[parent_id]

        # identify lower budget/fidelity to transfer information from
        lower_budget, num_configs = bracket.get_lower_budget_promotions(budget)

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
        finite_values = np.isfinite(self.de[lower_budget].fitness).any(axis=1)
        normalized_fitness = self.de[lower_budget].fitness.copy()

        if finite_values.any():
            normalized_fitness[finite_values] = normalize(normalized_fitness[finite_values], axis=0, norm='max')

        if self.mo_strategy == 'NSGA-II':
            mutation_pop_idx = self._select_best_config_mo_cd(normalized_fitness)[:num_configs]
        elif self.mo_strategy == 'EPSNET':
            mutation_pop_idx = self._select_best_config_epsnet(normalized_fitness)[:num_configs]
        else:
            raise ValueError(f'Unknown mo strategy, {self.mo_strategy}')

        mutation_pop = self.de[lower_budget].population[mutation_pop_idx]

        # generate mutants from previous budget subpopulation or global population
        if len(mutation_pop) < self.de[budget]._min_pop_size:
            filler = self.de[budget]._min_pop_size - len(mutation_pop) + 1
            sub_pop, _ = self._concat_all_budget_pop()
            new_pop = self.de[budget]._init_mutant_population(
                pop_size=filler, population=np.array(sub_pop))
            mutation_pop = np.concatenate((mutation_pop, new_pop))
        mutant = self.de[budget].mutation(
            current=target, alt_pop=mutation_pop
        )
        # perform crossover with selected parent
        config = self.de[budget].crossover(target=target, mutant=mutant)
        config = self.de[budget].boundary_check(config)
        return config, parent_id

    # ''' This function checks the fitness of parent and evaluated config and replace parent only
    # if its fitness(evaluated by NDS and hypervolume) is greater than the parent, if both lies on the same front,
    # replace least hv contributor'''

    def check_fitness(self, current_fitness, global_parent_id, parent_id, budget, config):
        pop, fit = self._concat_all_budget_pop()
        fit.extend([current_fitness])
        pop.extend([config])
        curr_idx = len(fit) - 1
        parent_idx = global_parent_id
        logger.debug("parent idx:{}", parent_idx)
        fitness = np.array([np.array(x) for x in fit])
        index_list = np.array(list(range(len(fit))))
        fronts, _, index_return_list = multi_obj_util.nDS_index(np.array(fitness), index_list)
        logger.debug("fronts:{}", fronts)
        logger.debug("index return list:{}", index_return_list)

        for idx, front_index in enumerate(index_return_list):
            front_index = front_index
            if curr_idx not in front_index and parent_idx not in front_index:
                continue
            if curr_idx in front_index and parent_idx in front_index:
                logger.debug("fitness:{}", fitness)
                # Removing unevaluated configs
                evaluated_configs_fitness = [item.tolist() for item in fitness if np.inf not in item]
                logger.debug("evaluated config fitness:{}", evaluated_configs_fitness)
                eval_index_list = np.array(list(range(len(evaluated_configs_fitness))))
                eval_fronts, _, eval_index_return_list = multi_obj_util.nDS_index(np.array(evaluated_configs_fitness),
                                                                                  eval_index_list,
                                                                                  return_front_indices=True)

                idx = multi_obj_util.minHV3D(eval_fronts[-1])
                lowest_hv_fitness = eval_fronts[-1][idx]
                idx = np.where(np.all(fitness == lowest_hv_fitness, axis=1))[0][0]
                # idx = fitness.tolist().index(lowest_hv_fitness)
                logger.debug("index returned:{}", idx)
                if idx == curr_idx:
                    return
                budget, parent_id = self._get_info_by_global_parent_id(idx)
                logger.debug("parent id :{},budgte:{}", parent_id, budget)
                self.de[budget].population[parent_id] = config
                self.de[budget].fitness[parent_id] = np.array(current_fitness)
                return

            elif curr_idx in front_index and parent_idx not in front_index:
                ''' Updating the population by replacing parent with child
                '''
                self.de[budget].population[parent_id] = config
                self.de[budget].fitness[parent_id] = np.array(current_fitness)
                return
            else:
                logger.debug("choose parent")
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
            self.cumulated_costs += cost
            info = run_info["info"] if "info" in run_info else dict()
            budget, parent_id = run_info["budget"], run_info["parent_id"]
            config = run_info["config"]
            bracket_id = run_info["bracket_id"]
            global_parent_id = run_info["global_parent_id"]
            for bracket in self.active_brackets:
                if bracket.bracket_id == bracket_id:
                    # bracket job complete
                    bracket.complete_job(budget)  # IMPORTANT to perform synchronous SH
            self.check_fitness(fitness, global_parent_id, parent_id, budget, config)
            self._update_pareto()

            # book-keeping
            if len(self.runtime) > 0:
                runtime = self.runtime[-1] + cost
            else:
                runtime = cost
            self._update_trackers(
                runtime=runtime,
                history=(config.tolist(), fitness, float(cost), float(budget), info)
            )

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
            verbose=True, debug=False, save_intermediate=True, save_history=True, total_wallclock_cost=None, **kwargs):
        """ Main interface to run optimization by DEHB

        This function waits on workers and if a worker is free, asks for a configuration and a
        budget to evaluate on and submits it to the worker. In each loop, it checks if a job
        is complete, fetches the results, carries the necessary processing of it asynchronously
        to the worker computations.

        The duration of the DEHB run can be controlled by specifying one of 3 parameters. If more
        than one are specified, DEHB selects only one in the priority order (high to low):
        1) Number of function evaluations (fevals)
        2) Number of Successive Halving brackets run under Hyperband (brackets)
        3) Total computational cost (in seconds) aggregated by all function evaluations (total_wallclock_cost)
        4) Total computational cost (in seconds) returned by the objective function. Might be simulated costs. (total_cost)
        """
        self._init_subpop()
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
                if self._is_run_budget_exhausted(fevals, brackets, total_cost, total_wallclock_cost):
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
                            self._verbosity_runtime(fevals, brackets, total_cost, total_wallclock_cost=None, )
                            self.logger.debug(
                                "Evaluating a configuration with budget {} under "
                                "bracket ID {}".format(budget, job_info['bracket_id'])
                            )
                        self._verbosity_debug()

                self._fetch_results_from_workers()
                if save_history and self.history is not None:
                    self._save_history()
                self.clean_inactive_brackets()
            # logger.debug("constraint tracker is {}", self.history)
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
            return np.array(self.runtime), np.array(self.history, dtype=object)

        except(KeyboardInterrupt, Exception) as err:
            logger.error("exception caught:{}", err)
            self.save_results()
            traceback.print_exc()
            return np.array(self.runtime), np.array(self.history, dtype=object)
