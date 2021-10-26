import logging
import os
import json
import pickle
import time
import ConfigSpace as CS
import numpy as np
import argparse
import sys
import random
from loguru import logger
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant
from sklearn.model_selection import StratifiedKFold
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision.datasets import ImageFolder
from distributed import Client
from cnn import torchModel
import DEHB.dehb.optimizers.modehb as mo  # import MODE

logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}


def get_optimizer_and_crit(cfg,model_param,lr):
    if cfg['optimizer'] == 'SGD':
        model_optimizer = torch.optim.SGD
        optimizer = model_optimizer(model_param, lr=lr, momentum=cfg['momentum'])
    elif cfg['optimizer'] == 'Adam':
        model_optimizer = torch.optim.Adam
        optimizer = model_optimizer(model_param,lr=lr, weight_decay=cfg['weight_decay'])
    else:
        model_optimizer = torch.optim.AdamW
        optimizer = model_optimizer(model_param,lr=lr, weight_decay=cfg['weight_decay'])

    train_criterion = torch.nn.CrossEntropyLoss
    return optimizer, train_criterion


def data_aug_list(cfg, size, seed):
    random.seed(42)
    aug_list = []
    if cfg['"resize"'] == True:
        aug_list.append(transforms.Resize(size))
    if cfg['horizontal_flip'] == True:
        aug_list.append(transforms.RandomHorizontalFlip(p=cfg['horizontal_flip_prob']))
    if cfg['random_crop'] == True:
        aug_list.append(transforms.RandomCrop(size, pad_if_needed=True))
    if cfg['rotate'] == True:
        aug_list.append(transforms.RandomRotation(degrees=(-180,180)))
    return aug_list


# Target Algorithm
# The signature of the function determines what arguments are passed to it
# i.e., budget is passed to the target algorithm if it is present in the signature
def cnn_from_cfg(cfg, seed, budget, run=1, **kwargs):
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    lr = cfg['learning_rate_init'] if cfg['learning_rate_init'] else 0.001
    batch_size = cfg['batch_size'] if cfg['batch_size'] else 200
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_width = 16
    img_height = 16
    output_path = kwargs["output_path"]
    # image size
    input_shape = (3, img_width, img_height)

    aug_list = data_aug_list(cfg, img_width, seed)
    if aug_list is None or len(aug_list) is 0:
        # You can add any preprocessing/data augmentation you want here
        data_augmentations = transforms.ToTensor()
    else:
        aug_list.append(transforms.ToTensor())
        data_augmentations = transforms.Compose(aug_list)

    data_dir = kwargs["train"]

    data = ImageFolder(os.path.join(data_dir, "train"), transform=data_augmentations)
    constraint_model_size = kwargs["constraint_model_size"]
    constraint_precision = kwargs["constraint_precision"]
    # instantiate optimizer

    num_epochs = int(np.ceil(budget))

    # Train the model
    score = []
    score_precisions = []

    # returns the cross validation accuracy
    cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)  # to make CV splits consistent
    # for train_idx, valid_idx in cv.split(data, data.targets):

    train_sets = []
    val_sets = []
    start = time.time()
    num_classes = len(data.classes)
    logging.info("num classes:{}".format(num_classes))

    for train_idx, valid_idx in cv.split(data, data.targets):
        train_sets.append(Subset(data, train_idx))
        val_sets.append(Subset(data, valid_idx))

    for train_set, val_set in zip(train_sets, val_sets):
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(dataset=val_set,
                                batch_size=batch_size,
                                shuffle=False)
        model = torchModel(cfg,
                       input_shape=input_shape,
                       num_classes=num_classes).to(device)
        total_model_params = np.sum(p.numel() for p in model.parameters())

        optimizer, train_criterion = get_optimizer_and_crit(cfg,model.parameters(),lr)

        # instantiate training criterion
        train_criterion = train_criterion().to(device)

        logging.info('Generated Network:')
        summary(model, input_shape,
            device='cuda' if torch.cuda.is_available() else 'cpu')

        for epoch in range(num_epochs):
            logging.info('#' * 50)
            logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
            train_score, train_loss = model.train_fn(optimizer, train_criterion, train_loader, device)
            val_score, _, val_score_precision = model.eval_fn(val_loader, device)
            logging.info('Test accuracy %f', val_score)
            logging.info('Train accuracy %f', train_score)
        score_accuracy_top3, _, score_precision = model.eval_fn(val_loader, device)
        logging.info('Test accuracy :{}'.format(score_accuracy_top3))
        logging.info("score precision:{}".format(score_precision))
        score.append(score_accuracy_top3)
        score_precisions.append(score_precision)
    cost = time.time() - start
    # instantiate training criterion
    acc = 1 - np.mean(score)  # to minimize
    precision = np.mean(score_precisions)
    with open(output_path + 'dehb_run.json', 'a+')as f:
        json.dump({'configuration': dict(cfg), 'error': acc, 'top3': 1 - acc,
                   'precision': precision,
                   'n_params': total_model_params, 'num_epochs': num_epochs}, f)

        f.write("\n")
    #Writing to two files based on constraints and accuracy threshold as for a few configuration, accuracy is very close
    #to the optimum and such configurations can also give some insights
    if (total_model_params <= constraint_model_size and precision >= constraint_precision):
        with open(output_path + 'constraint_satisfied_cfg.json', 'a+')as f:
            json.dump({'configuration': dict(cfg), 'error': acc, 'top3': 1 - acc,
                       'precision': precision,
                       'n_params': total_model_params, 'num_epochs': num_epochs}, f)
            f.write("\n")
    if (acc <= 0.2 and total_model_params <= constraint_model_size and precision >= constraint_precision):
        with open(output_path + 'opt_cfg.json', 'a+')as f:
            json.dump({'configuration': dict(cfg), 'error': acc, 'top3': 1 - acc,
                       'precision': precision,
                       'n_params': total_model_params, 'num_epochs': num_epochs}, f)
            f.write("\n")

    return ({"cost": cost,
             "fitness": [total_model_params, -precision, acc]})  # Because minimize!


def dehb_setup(args, cs, dimensions):
    # Some insights into Dask interfaces to DEHB and handling GPU devices for parallelism:
    # * if args.scheduler_file is specified, args.n_workers need not be specifed --- since
    #    args.scheduler_file indicates a Dask client/server is active
    # * if args.scheduler_file is not specified and args.n_workers > 1 --- the DEHB object
    #    creates a Dask client as at instantiation and dies with the associated DEHB object
    # * if args.single_node_with_gpus is True --- assumes that all GPU devices indicated
    #    through the environment variable "CUDA_VISIBLE_DEVICES" resides on the same machine

    # Dask checks and setups
    single_node_with_gpus = args.single_node_with_gpus
    if args.scheduler_file is not None and os.path.isfile(args.scheduler_file):
        client = Client(scheduler_file=args.scheduler_file)
        # explicitly delegating GPU handling to Dask workers defined
        single_node_with_gpus = False
    else:
        client = None
    ###########################
    # DEHB optimisation block #
    ###########################
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'micro17flower')
    np.random.seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    output_path = args.output_path + "_" + str(args.seed) + "_" + str(args.run)  + '/'
    os.makedirs(output_path, exist_ok=True)
    modehb = mo.MODEHB(f=cnn_from_cfg, cs=cs, dimensions=dimensions, min_budget=args.min_budget,
                       max_budget=args.max_budget, eta=args.eta,
                       constraint_model_size=args.constraint_max_model_size,
                       constraint_min_precision=args.constraint_min_precision,
                       output_path=output_path,
                       # if client is not None and of type Client, n_workers is ignored
                       # if client is None, a Dask client with n_workers is set up
                       client=client, n_workers=args.n_workers, seed=args.seed)
    runtime, history, pareto_pop, pareto_fit = modehb.run(total_cost=args.runtime, verbose=args.verbose,
                                                          # arguments below are part of **kwargs shared across workers
                                                          train=data_dir,
                                                          constraint_model_size=args.constraint_max_model_size,
                                                          constraint_precision=args.constraint_min_precision,
                                                          single_node_with_gpus=single_node_with_gpus,
                                                          output_path=output_path,
                                                          device=device)

    log_suffix = time.strftime("%x %X %Z")
    log_suffix = log_suffix.replace("/", '-').replace(":", '-').replace(" ", '_')
    with open(os.path.join(output_path, "history_{}.pkl".format(log_suffix)), "wb") as f:
        pickle.dump(history, f)

def get_configspace(seed=1):
    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.
    cs = CS.ConfigurationSpace(seed)

    n_conf_layer = UniformIntegerHyperparameter("n_conv_layers", 1, 3, default_value=3)
    n_conf_layer_0 = UniformIntegerHyperparameter("n_channels_conv_0", 1, 2048, default_value=512)
    n_conf_layer_1 = UniformIntegerHyperparameter("n_channels_conv_1", 1, 2048, default_value=512)
    n_conf_layer_2 = UniformIntegerHyperparameter("n_channels_conv_2", 1, 2048, default_value=512)

    n_fc_layers = UniformIntegerHyperparameter("n_fc_layers", 1, 3, default_value=3)
    n_channels_fc_0 = UniformIntegerHyperparameter("n_channels_fc_0", 1, 1024, default_value=128, log=True)
    n_channels_fc_1 = UniformIntegerHyperparameter("n_channels_fc_1", 1, 1024, default_value=64, log=True)
    n_channels_fc_2 = UniformIntegerHyperparameter("n_channels_fc_2", 1, 1024, default_value=32, log=True)
    learning_rate_init = UniformFloatHyperparameter("learning_rate_init", lower=1e-6, upper=1e-2, log=True)

    # Kernel Size
    ks = CategoricalHyperparameter("kernel_size", choices=[7, 5, 3], default_value=5)

    # Use Batch Normalization
    bn = CategoricalHyperparameter("batch_norm", choices=[False, True], default_value=False)

    # Batch size
    bs = UniformIntegerHyperparameter('batch_size', 32, 512, default_value=128, log=True)

    # Dropout rate
    dropout_rate = UniformFloatHyperparameter('dropout_rate', lower=0, upper=0.8, log=False, default_value=0.2)

    # Weight Decay
    weight_decay = UniformFloatHyperparameter('weight_decay', lower=1e-6, upper=0.01, log=True, default_value=0.001)

    # Optimizer
    optimizer = CategoricalHyperparameter("optimizer", choices=['AdamW', 'Adam', 'SGD'], default_value='AdamW')

    # momentum
    momentum = UniformFloatHyperparameter('momentum', lower=0, upper=1, log=False, default_value=0.9)

    #data augmentation parameters
    resize = CategoricalHyperparameter("resize", choices=[False, True], default_value=False)

    random_crop = CategoricalHyperparameter("random_crop", choices=[False, True], default_value=False)

    horizontal_flip = CategoricalHyperparameter("horizontal_flip", choices=[False, True], default_value=False)

    horizontal_flip_prob = UniformFloatHyperparameter("horizontal_flip_prob", lower=0.1, upper=1, log=False, default_value=0.5)

    rotate = CategoricalHyperparameter("rotate", choices=[False, True], default_value=False)

    # Global Avg Pooling
    ga = CategoricalHyperparameter("global_avg_pooling", choices=[False, True], default_value=True)
    cs.add_hyperparameters([n_conf_layer, n_conf_layer_0, n_conf_layer_1, n_conf_layer_2,
                            learning_rate_init, optimizer, ks, bn, bs, ga, n_channels_fc_0, n_channels_fc_1,
                            n_channels_fc_2, n_fc_layers, dropout_rate,
                            weight_decay
                            ,resize,random_crop,horizontal_flip,horizontal_flip_prob
                            ,momentum
                            ,rotate
                            ])

    # Add conditions to restrict the hyperparameter space
    use_linear_layer_2 = CS.conditions.InCondition(n_channels_fc_2, n_fc_layers, [3])
    use_linear_layer_1 = CS.conditions.InCondition(n_channels_fc_1, n_fc_layers, [2, 3])
    use_linear_layer_0 = CS.conditions.InCondition(n_channels_fc_0, n_fc_layers, [1, 2, 3])
    use_conf_layer_2 = CS.conditions.InCondition(n_conf_layer_2, n_conf_layer, [3])
    use_conf_layer_1 = CS.conditions.InCondition(n_conf_layer_1, n_conf_layer, [2, 3])
    use_horizontal_flip_prob = CS.conditions.InCondition(horizontal_flip_prob, horizontal_flip, [True])
    use_momentum = CS.conditions.InCondition(momentum, optimizer, ['SGD'])
    # Add  multiple conditions on hyperparameters at once:
    cs.add_conditions([use_conf_layer_2, use_conf_layer_1,
                       use_linear_layer_0, use_linear_layer_1, use_linear_layer_2,
                       use_horizontal_flip_prob,
                       use_momentum])
    return cs


def input_arguments():
    parser = argparse.ArgumentParser(description='Optimizing MNIST in PyTorch using DEHB.')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 123)')
    parser.add_argument('--refit_training', action='store_true', default=False,
                        help='Refit with incumbent configuration on full training data and budget')
    parser.add_argument('--min_budget', type=float, default=10,
                        help='Minimum budget (epoch length)')
    parser.add_argument('--max_budget', type=float, default=50,
                        help='Maximum budget (epoch length)')
    parser.add_argument('--run', type=int, default=1,
                        help='run number')
    parser.add_argument('--eta', type=int, default=3,
                        help='Parameter for Hyperband controlling early stopping aggressiveness')
    parser.add_argument('--output_path', type=str, default="/content/drive/MyDrive/run/",
                        help='Directory for DEHB to write logs and outputs')
    parser.add_argument('--scheduler_file', type=str, default=None,
                        help='The file to connect a Dask client with a Dask scheduler')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of CPU workers for DEHB to distribute function evaluations to')

    parser.add_argument('--single_node_with_gpus', default=False, action="store_true",
                        help='If True, signals the DEHB run to assume all required GPUs are on '
                             'the same node/machine. To be specified as True if no client is '
                             'passed and n_workers > 1. Should be set to False if a client is '
                             'specified as a scheduler-file created. The onus of GPU usage is then'
                             'on the Dask workers created and mapped to the scheduler-file.')
    parser.add_argument('-v', '--verbose',
                        default='INFO',
                        choices=['INFO', 'DEBUG'],
                        help='verbosity')
    parser.add_argument('--runtime', type=float, default=86400,
                        help='initialize population using SMAC on cheap evaluation: model size')
    parser.add_argument('-s', "--constraint_max_model_size",
                        default=2e7,
                        help="maximal model size constraint",
                        type=int)
    parser.add_argument('-p', "--constraint_min_precision",
                        default=0.42,
                        help='minimal constraint constraint',
                        type=float)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    args = input_arguments()
    constraint_model_size = args.constraint_max_model_size
    constraint_precision = args.constraint_min_precision

    logger = logging.getLogger("AutoML Optimizer")
    logging.basicConfig(level=logging.INFO)

    # Get configuration space
    cs = get_configspace(args.seed)
    dimensions = len(cs.get_hyperparameters())
    dehb_setup(args, cs, dimensions)

