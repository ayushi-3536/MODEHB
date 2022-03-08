#import nas_201_api
import torch
import numpy as np
#from nas_201_api import NASBench201API as API
from ax import Metric
from ax.core.search_space import SearchSpace
from ax.core.objective import MultiObjective
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from baselines import MultiObjectiveSimpleExperiment

import ConfigSpace as CS
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import CategoricalHyperparameter


class NASSearchSpace(CS.ConfigurationSpace):

    def __init__(self):
        super(NASSearchSpace, self).__init__()

        # Convolution


        # p1 = UniformIntegerHyperparameter("p1", 0, 4, default_value=0, log=False)
        # p2 = UniformIntegerHyperparameter("p2", 0, 4, default_value=0, log=False)
        # p3 = UniformIntegerHyperparameter("p3", 0, 4, default_value=0, log=False)
        # p4 = UniformIntegerHyperparameter("p4", 0, 4, default_value=0, log=False)
        # p5 = UniformIntegerHyperparameter("p5", 0, 4, default_value=0, log=False)
        # p6 = UniformIntegerHyperparameter("p6", 0, 4, default_value=0, log=False)
        #

        p1 = CategoricalHyperparameter("1<-0", choices=['none', 'skip_connect', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3'], default_value='none')
        p2 = CategoricalHyperparameter("2<-0", choices=['none', 'skip_connect', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3'], default_value='none')
        p3 = CategoricalHyperparameter("2<-1", choices=['none', 'skip_connect', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3'], default_value='none')
        p4 = CategoricalHyperparameter("3<-0", choices=['none', 'skip_connect', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3'], default_value='none')
        p5 = CategoricalHyperparameter("3<-1", choices=['none', 'skip_connect', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3'], default_value='none')
        p6 = CategoricalHyperparameter("3<-2", choices=['none', 'skip_connect', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3'], default_value='none')

        self.not_mutables = ['budget']
        self.not_mutables = ['id']
        self.add_hyperparameters([p1, p2, p3, p4, p5, p6])


    def as_uniform_space(self):


        p1 = self.get_hyperparameter('1<-0')
        p2 = self.get_hyperparameter('2<-0')
        p3 = self.get_hyperparameter('2<-1')
        p4 = self.get_hyperparameter('3<-0')
        p5 = self.get_hyperparameter('3<-1')
        p6 = self.get_hyperparameter('3<-2')

        cs = CS.ConfigurationSpace()

        cs.add_hyperparameters([p1, p2, p3, p4, p5, p6])

        return cs

    def as_ax_space(self):
        from ax import ParameterType, RangeParameter, FixedParameter, ChoiceParameter, SearchSpace

        i = FixedParameter('id', ParameterType.STRING, 'dummy')
        p1 = ChoiceParameter(name='1<-0', parameter_type=ParameterType.STRING, values=['none','skip_connect', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3'])
        p2 = ChoiceParameter(name='2<-0', parameter_type=ParameterType.STRING, values=['none','skip_connect', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3'])
        p3 = ChoiceParameter(name='2<-1', parameter_type=ParameterType.STRING, values=['none','skip_connect', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3'])
        p4 = ChoiceParameter(name='3<-0', parameter_type=ParameterType.STRING, values=['none','skip_connect', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3'])
        p5 = ChoiceParameter(name='3<-1', parameter_type=ParameterType.STRING, values=['none','skip_connect', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3'])
        p6 = ChoiceParameter(name='3<-2', parameter_type=ParameterType.STRING, values=['none','skip_connect', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3'])
        b = FixedParameter('budget', ParameterType.INT, 199)

       # b = FixedParameter('budget', ParameterType.INT, 25)



        return SearchSpace(
            parameters=[p1,p2,p3,p4,p5,p6,b,i],
        )


    def sample_hyperparameter(self, hp):
        if not self.is_mutable_hyperparameter(hp):
            raise Exception("Hyperparameter {} is not mutable and must be fixed".format(hp))
        return self.get_hyperparameter(hp).sample(self.random)

    def is_mutable_hyperparameter(self, hp):
        return hp not in self.not_mutables


