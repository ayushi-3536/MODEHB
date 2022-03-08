"""A common search space for all the experiments
"""

import ConfigSpace as CS
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import CategoricalHyperparameter


class CustomSearchSpace(CS.ConfigurationSpace):
    # lr = ag.space.Real(lower=1, upper=50, log=True),
    # dropout = ag.space.Real(lower=0, upper=0.99),
    # batch_size = ag.space.Int(
    #     lower=BATCH_SIZE_LOWER, upper=BATCH_SIZE_UPPER),
    # clip = ag.space.Real(lower=0.1, upper=2),
    # lr_factor = ag.space.Real(lower=1, upper=100, log=True),
    # emsize = ag.space.Int(lower=32, upper=1024)
    def __init__(self):
        super(CustomSearchSpace, self).__init__()


        # Learning Rate
        lr = UniformFloatHyperparameter('lr', 1, 50, default_value=5, log=True)
        lr_factor = UniformFloatHyperparameter('lr_factor', 1, 100, default_value=50, log=True)


        dropout = UniformFloatHyperparameter('dropout', 0, 0.99, default_value=0.99)
        clip = UniformFloatHyperparameter('clip', 0.1, 2, default_value=0.99)

        batch_size = UniformIntegerHyperparameter('batch_size', 8, 256, default_value=128)

        emsize = UniformIntegerHyperparameter('emsize', 32, 1024, default_value=128)

        self.not_mutables = []
        self.add_hyperparameters([lr, lr_factor, dropout, clip, batch_size,emsize])


    def as_uniform_space(self):
        # Learning Rate
        lr = self.get_hyperparameter('lr')
        lr_factor = self.get_hyperparameter('lr_factor')
        dropout = self.get_hyperparameter('dropout')
        clip = self.get_hyperparameter('clip')
        batch_size = self.get_hyperparameter('batch_size')
        emsize = self.get_hyperparameter('emsize')
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([ lr,lr_factor,dropout,clip,batch_size,emsize ])
        return cs

    def as_ax_space(self):
        from ax import ParameterType, RangeParameter, FixedParameter, ChoiceParameter, SearchSpace



        # Learning Rate
        lr =  RangeParameter('lr', ParameterType.FLOAT, 1, 50, True)
        lr_factor = RangeParameter('lr_factor', ParameterType.FLOAT, 1, 100, True)
        dropout = RangeParameter('dropout', ParameterType.FLOAT, 0, 0.99)
        clip = RangeParameter('clip', ParameterType.FLOAT, 0.1, 2)

        # Batch size
        batch_size = RangeParameter('batch_size', ParameterType.INT, 8,256)
        emsize = RangeParameter('emsize', ParameterType.INT, 32, 1024)

        b = FixedParameter('budget', ParameterType.INT, 81)

        i = FixedParameter('id', ParameterType.STRING, 'dummy')

        return SearchSpace(
            parameters=[lr,lr_factor,dropout,clip,batch_size,emsize, b, i],
        )


    def sample_hyperparameter(self, hp):
        if not self.is_mutable_hyperparameter(hp):
            raise Exception("Hyperparameter {} is not mutable and must be fixed".format(hp))
        return self.get_hyperparameter(hp).sample(self.random)

    def is_mutable_hyperparameter(self, hp):
        return hp not in self.not_mutables
