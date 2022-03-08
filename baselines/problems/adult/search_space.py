"""A common search space for all the experiments
"""

import ConfigSpace as CS
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import CategoricalHyperparameter


class CustomSearchSpace(CS.ConfigurationSpace):

    def __init__(self):
        super(CustomSearchSpace, self).__init__()

        # Dense
        n_fc_l = UniformIntegerHyperparameter("n_fc_l", 1, 4, default_value=3)
        n_fc_0 = UniformIntegerHyperparameter("n_fc_0", 2, 32, default_value=16, log=True)
        n_fc_1 = UniformIntegerHyperparameter("n_fc_1", 2, 32, default_value=16, log=True)
        n_fc_2 = UniformIntegerHyperparameter("n_fc_2", 2, 32, default_value=16, log=True)
        n_fc_3 = UniformIntegerHyperparameter("n_fc_3", 2, 32, default_value=16, log=True)

        # Alpha
        alpha = UniformFloatHyperparameter('alpha', 0.000001, 0.1, default_value=0.01, log=True)

        # Learning Rate
        lr = UniformFloatHyperparameter('lr', 0.000001, 0.01, default_value=0.001, log=True)

        # Beta1
        beta_1 = UniformFloatHyperparameter('beta_1', 0.001, 0.99, default_value=0.001, log=True)

        # Beta2
        beta_2 = UniformFloatHyperparameter('beta_2', 0.001, 0.99, default_value=0.001, log=True)

        # tol
        tol = UniformFloatHyperparameter('tol', 0.00001, 0.01, default_value=0.001, log=True)

        # Conditions

        cond1 = CS.conditions.InCondition(n_fc_3, n_fc_l, [4])
        cond2 = CS.conditions.InCondition(n_fc_2, n_fc_l, [3,4])
        cond3 = CS.conditions.InCondition(n_fc_1, n_fc_l, [2, 3,4])
        cond4 = CS.conditions.InCondition(n_fc_0, n_fc_l, [1, 2, 3,4])

        self.not_mutables = ['n_fc_l']

        self.add_hyperparameters([n_fc_l, n_fc_0, n_fc_1, n_fc_2,n_fc_3])
        self.add_hyperparameters([lr,alpha,beta_2,beta_1,tol])
        self.add_conditions([cond1, cond2, cond3, cond4])

    def as_uniform_space(self):



        # Dense
        n_fc_l = self.get_hyperparameter('n_fc_l')
        n_fc_0 = self.get_hyperparameter('n_fc_0')
        n_fc_1 = self.get_hyperparameter('n_fc_1')
        n_fc_2 = self.get_hyperparameter('n_fc_2')
        n_fc_3 = self.get_hyperparameter('n_fc_3')

        # Learning Rate
        lr = self.get_hyperparameter('lr')

        alpha = self.get_hyperparameter('alpha')
        beta_1 = self.get_hyperparameter('beta_1')
        beta_2 = self.get_hyperparameter('beta_2')
        tol = self.get_hyperparameter('tol')



        cond1 = CS.conditions.InCondition(n_fc_3, n_fc_l, [4])
        cond2 = CS.conditions.InCondition(n_fc_2, n_fc_l, [3,4])
        cond3 = CS.conditions.InCondition(n_fc_1, n_fc_l, [2, 3,4])
        cond4 = CS.conditions.InCondition(n_fc_0, n_fc_l, [1, 2, 3,4])

        cs = CS.ConfigurationSpace()

        cs.add_hyperparameters([n_fc_l, n_fc_0, n_fc_1, n_fc_2,n_fc_3])
        cs.add_hyperparameters([lr,alpha,beta_2,beta_1,tol])
        cs.add_conditions([cond1, cond2, cond3, cond4])
        return cs

    def as_ax_space(self):
        from ax import ParameterType, RangeParameter, FixedParameter, ChoiceParameter, SearchSpace

        # Dense
        n_fc_l = RangeParameter('n_fc_l', ParameterType.INT, 1, 4)
        n_fc_0 = RangeParameter('n_fc_0', ParameterType.INT, 2, 32, True)
        n_fc_1 = RangeParameter('n_fc_1', ParameterType.INT, 2, 32, True)
        n_fc_2 = RangeParameter('n_fc_2', ParameterType.INT, 2, 32, True)
        n_fc_3 = RangeParameter('n_fc_3', ParameterType.INT, 2, 32, True)

        # Learning Rate
        lr =  RangeParameter('lr', ParameterType.FLOAT, 0.000001, 0.001, True)

        #alpha
        alpha = RangeParameter('alpha', ParameterType.FLOAT, 0.000001, 0.01, True)

        #Beta
        beta_1 = RangeParameter('beta_1', ParameterType.FLOAT, 0.001, 0.99, True)
        beta_2 = RangeParameter('beta_2', ParameterType.FLOAT, 0.001, 0.99, True)

        #tol
        tol = RangeParameter('tol', ParameterType.FLOAT, 0.00001, 0.001, True)


        b = FixedParameter('budget', ParameterType.INT, 200)

        i = FixedParameter('id', ParameterType.STRING, 'dummy')

        return SearchSpace(
            parameters=[n_fc_l, n_fc_0, n_fc_1, n_fc_2,n_fc_3, lr, alpha,beta_1,beta_2,tol, b, i],
        )


    def sample_hyperparameter(self, hp):
        if not self.is_mutable_hyperparameter(hp):
            raise Exception("Hyperparameter {} is not mutable and must be fixed".format(hp))
        return self.get_hyperparameter(hp).sample(self.random)

    def is_mutable_hyperparameter(self, hp):
        return hp not in self.not_mutables