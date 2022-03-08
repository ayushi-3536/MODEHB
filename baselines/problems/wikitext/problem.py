from ax import Metric
from ax import MultiObjective
from ax import ObjectiveThreshold
from ax import MultiObjectiveOptimizationConfig

from baselines import MultiObjectiveSimpleExperiment
from .wikinet import run_wikitext2
from .wikinetsetup import objective_func
from .search_space import CustomSearchSpace

def get_wikitext_ppl_score(name=None):
    # 'perplexity': (perplexity, 0.0),
    # 'val_acc': (val_acc, 0.0),
    # 'val_err': (1 - val_acc, 0.0),
    # 'neg_log_perplexity': (neg_loss_perplexity, 0.0),
    # 'prediction_time': (prediction_time, 0.0),
    # 'neg_prediction_time': (neg_prediction_time, 0.0),
    # 'elapsed_time': (elapsed_time, 0.0)

    perplexity = Metric('perplexity', True)
    val_acc = Metric('val_acc')
    val_error = Metric('val_error', True)
    log_perplexity = Metric('log_perplexity',True)
    neg_log_perplexity = Metric('neg_log_perplexity')
    prediction_time = Metric('prediction_time', True)
    neg_prediction_time = Metric('neg_prediction_time', False)
    elapsed_time = Metric('elapsed_time', True)

    objective = MultiObjective([log_perplexity, val_error])
    thresholds = [
        ObjectiveThreshold(log_perplexity, 10.0),
        ObjectiveThreshold(val_error, 1.0)
    ]
    optimization_config = MultiObjectiveOptimizationConfig(
        objective=objective,
        objective_thresholds=thresholds
    )

    return MultiObjectiveSimpleExperiment(
        name=name,
        search_space=CustomSearchSpace().as_ax_space(),
        eval_function=objective_func,
        optimization_config=optimization_config,
        extra_metrics=[val_acc,perplexity, neg_log_perplexity,prediction_time,neg_prediction_time,elapsed_time]#, runtime, eval_runtime]
    )
