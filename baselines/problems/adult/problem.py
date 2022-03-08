from ax import Metric
from ax import MultiObjective
from ax import ObjectiveThreshold
from ax import MultiObjectiveOptimizationConfig

from baselines import MultiObjectiveSimpleExperiment
from .adultnet import evaluate_network
from .search_space import CustomSearchSpace

def get_acc_dsp(name=None):

    dsp = Metric('dsp', True)
    deo = Metric('deo', True)
    dfp = Metric('dfp', True)
    train_acc = Metric('train_acc', True)
    test_acc = Metric('test_acc', True)
    test_error = Metric('test_error', True)


    #runtime = Metric('total_runtime', True)
    #eval_runtime = Metric('eval_runtime', True)

    objective = MultiObjective([test_error, dsp])
    thresholds = [
        ObjectiveThreshold(test_error, 1.0),
        ObjectiveThreshold(dsp, 1.0)
    ]
    optimization_config = MultiObjectiveOptimizationConfig(
        objective=objective,
        objective_thresholds=thresholds
    )

    return MultiObjectiveSimpleExperiment(
        name=name,
        search_space=CustomSearchSpace().as_ax_space(),
        eval_function=evaluate_network,
        optimization_config=optimization_config,
        extra_metrics=[deo,dfp,train_acc,test_acc]#, runtime, eval_runtime]
    )

def get_acc_dsp_deo(name=None):

    dsp = Metric('dsp', True)
    deo = Metric('deo', True)
    dfp = Metric('dfp', True)
    train_acc = Metric('train_acc', True)
    test_acc = Metric('test_acc', True)
    test_error = Metric('test_error', True)
    #runtime = Metric('total_runtime', True)
    #eval_runtime = Metric('eval_runtime', True)

    objective = MultiObjective([test_error, dsp,deo])
    thresholds = [
        ObjectiveThreshold(test_error, 1.0),
        ObjectiveThreshold(dsp, 1),
        ObjectiveThreshold(deo, 1),
    ]
    optimization_config = MultiObjectiveOptimizationConfig(
        objective=objective,
        objective_thresholds=thresholds
    )

    return MultiObjectiveSimpleExperiment(
        name=name,
        search_space=CustomSearchSpace().as_ax_space(),
        eval_function=evaluate_network,
        optimization_config=optimization_config,
        extra_metrics=[dfp,train_acc,test_acc]#, runtime, eval_runtime]
    )

def get_acc_dsp_deo_dfp(name=None):

    dsp = Metric('dsp', True)
    deo = Metric('deo', True)
    dfp = Metric('dfp', True)
    train_acc = Metric('train_acc', True)
    test_acc = Metric('test_acc', True)
    test_error = Metric('test_error', True)
    #runtime = Metric('total_runtime', True)
    #eval_runtime = Metric('eval_runtime', True)

    objective = MultiObjective([test_error, dsp,deo])
    thresholds = [
        ObjectiveThreshold(test_acc, 1.0),
        ObjectiveThreshold(dsp, 1),
        ObjectiveThreshold(deo, 1),
        ObjectiveThreshold(dfp, 1),
    ]
    optimization_config = MultiObjectiveOptimizationConfig(
        objective=objective,
        objective_thresholds=thresholds
    )

    return MultiObjectiveSimpleExperiment(
        name=name,
        search_space=CustomSearchSpace().as_ax_space(),
        eval_function=evaluate_network,
        optimization_config=optimization_config,
        extra_metrics=[train_acc,test_acc]#, runtime, eval_runtime]
    )
