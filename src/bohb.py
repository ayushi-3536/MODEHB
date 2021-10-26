"""
===========================
Optimization using BOHB
===========================
"""
import os
import numpy as np
from functools import partial
from smac.facade.smac_ac_facade import SMAC4AC
from smac.scenario.scenario import Scenario
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import sys
from cnn import torchModel

from loguru import logger

logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}


# Target Algorithm
# The signature of the function determines what arguments are passed to it
# i.e., budget is passed to the target algorithm if it is present in the signature
def cnn_from_cfg(cfg, data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'micro17flower')):
    """
        Creates an instance of the torch_model and fits the given data on it.
        This is the function-call we try to optimize. Chosen values are stored in
        the configuration (cfg).

        Parameters
        ----------
        cfg: Configuration (basically a dictionary)
            configuration chosen by smac
        seed: int or RandomState
            used to initialize the rf's random generator
        instance: str
            used to represent the instance to use (just a placeholder for this example)
        budget: float
            used to set max iterations for the MLP

        Returns
        -------
        float
    """
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_width = 16
    img_height = 16
    data_augmentations = transforms.ToTensor()
    data = ImageFolder(os.path.join(data_dir, "train"), transform=data_augmentations)
    input_shape = (3, img_width, img_height)

    model = torchModel(cfg,
                       input_shape=input_shape,
                       num_classes=len(data.classes)).to(device)
    total_model_params = np.sum(p.numel() for p in model.parameters())
    return total_model_params  # Because minimize!


#This function optimizes only for the model parameters
def bohb_optimization(cs, seed, pop_size, output_dir, constraints):
    # SMAC scenario object

    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative to runtime)
                         "wallclock-limit": 200,  # max duration to run the optimization (in seconds)
                         "ta_run_limit": 30,
                         "cs": cs,  # configuration space
                         "deterministic": True,
                         "output_dir": output_dir + 'smac/',
                         "evaluations": 25,
                         "limit_resources": True,  # Uses pynisher to limit memory and runtime
                         # Alternatively, you can also disable this.
                         # Then you should handle runtime and memory yourself in the TA
                         "memory_limit": 3072,  # adapt this to reasonable value for your hardware
                         })

    # To optimize, we pass the function to the SMAC-object

    smac = SMAC4AC(scenario=scenario, rng=np.random.RandomState(seed),
                   tae_runner=partial(cnn_from_cfg),
                   initial_design_kwargs={'n_configs_x_params': 1, 'max_config_fracs': .2})

    #  # Start optimization

    smac.solver.scenario.intensification_percentage = 0.5
    try:  # try finally used to catch any interupt
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent
    configs = smac.solver.runhistory.get_all_configs()
    logger.debug("len con:{}", len(configs))
    costs = []
    valid_configs = []
    for config in configs:
        cost = smac.solver.runhistory.get_cost(config)
        if (cost <= constraints[0]):
            valid_configs.append(config)
            costs.append(cost)
    logger.debug("config after:{}", valid_configs)
    logger.debug("costs:{}", costs)
    return valid_configs, costs  # [-pop_size:]
