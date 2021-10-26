import os
import argparse
import logging
import time
import numpy as np
import json
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold  # We use 3-fold stratified cross-validation
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision.datasets import ImageFolder

from cnn import torchModel

min_acc = 0


def main(model_config,
         data_dir,
         num_epochs,
         batch_size,
         learning_rate,
         model_optimizer,
         constraints,
         data_augmentations,
         save_model_str,
         use_test_data=True,
         train_criterion=torch.nn.CrossEntropyLoss):
    """
    Training loop for configurableNet.
    :param model_config: network config (dict)
    :param data_dir: dataset path (str)
    :param use_test_data: if we use a separate test dataset or crossvalidation
    :param num_epochs: (int)
    :param batch_size: (int)
    :param learning_rate: model optimizer learning rate (float)
    :param train_criterion: Which loss to use during training (torch.nn._Loss)
    :param model_optimizer: Which model optimizer to use during trainnig (torch.optim.Optimizer)
    :param constraints: Constraints that needs to be fulfilled, the order determines the degree of difficulty
    :param data_augmentations: List of data augmentations to apply such as rescaling.
        (list[transformations], transforms.Composition[list[transformations]], None)
        If none only ToTensor is used
    :return:
    """

    # Device configuration
    if constraints is None:
        constraints = OrderedDict([('n_params', 5e7), ('precision', 0.61)])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_width = 16
    img_height = 16
    data_augmentations = data_aug_list(model_config, img_width)
    print(batch_size)
    if data_augmentations is None or len(data_augmentations) is 0:
        # You can add any preprocessing/data augmentation you want here
        data_augmentations = transforms.ToTensor()
    else:
        data_augmentations.append(transforms.ToTensor())
        data_augmentations = transforms.Compose(data_augmentations)
    print(data_augmentations)
    # Load the dataset
    tv_data = ImageFolder(os.path.join(data_dir, 'train'), transform=data_augmentations)
    test_data = ImageFolder(os.path.join(data_dir, 'test'), transform=data_augmentations)

    cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)  # to make CV splits consistent

    train_sets = []
    val_sets = []
    if use_test_data:
        train_sets.append(tv_data)
        val_sets.append(test_data)
    else:
        for train_idx, valid_idx in cv.split(tv_data, tv_data.targets):
            train_sets.append(Subset(tv_data, train_idx))
            val_sets.append(Subset(tv_data, valid_idx))

    # instantiate training criterion
    train_criterion = train_criterion().to(device)
    scores_accuracy = []
    scores_precision = []

    num_classes = len(tv_data.classes)
    # image size
    input_shape = (3, img_width, img_height)
    for train_set, val_set in zip(train_sets, val_sets):
        # Make data batch iterable
        # Could modify the sampler to not uniformly random sample
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(dataset=val_set,
                                batch_size=batch_size,
                                shuffle=False)

        model = torchModel(model_config,
                           input_shape=input_shape,
                           num_classes=num_classes).to(device)

        # THIS HERE IS THE FIRST CONSTRAINT YOU HAVE TO SATISFY
        # THIS HERE IS THE FIRST CONSTRAINT YOU HAVE TO SATISFY
        # THIS HERE IS THE FIRST CONSTRAINT YOU HAVE TO SATISFY
        total_model_params = np.sum(p.numel() for p in model.parameters())
        print(learning_rate)
        print(model_optimizer)
        # instantiate optimizer
        optimizer = get_optimizer_and_crit(model_config, model.parameters(), learning_rate, model_optimizer)

        # Just some info for you to see the generated network.
        logging.info('Generated Network:')
        summary(model, input_shape,
                device='cuda' if torch.cuda.is_available() else 'cpu')

        # Train the model
        for epoch in range(num_epochs):
            logging.info('#' * 50)
            logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

            train_score, train_loss = model.train_fn(optimizer, train_criterion, train_loader, device)
            score, _, score_precision = model.eval_fn(val_loader, device)

            logging.info('Train accuracy %f', train_score)
            logging.info('Test accuracy %f', score)
        score_accuracy_top3, _, score_precision = model.eval_fn(val_loader, device)

        scores_accuracy.append(score_accuracy_top3)
        scores_precision.append(np.mean(score_precision))

        # if save_model_str:
        #     # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
        #     if os.path.exists(save_model_str):
        #         save_model_str += '_'.join(time.ctime())
        #     torch.save(model.state_dict(), save_model_str)

    # RESULTING METRIC
    # RESULTING METRIC
    # RESULTING METRIC
    optimized_metrics = {"n_params": total_model_params,
                         "precision": np.mean(scores_precision),
                         "top3_accuracy": np.mean(scores_accuracy)}
    check = True
    for constraint_name in constraints:
        if constraint_name == 'model_size':
            # HERE IS THE CONSTRAINT THAT MUST BE SATISFIED
            assert optimized_metrics[constraint_name] <= constraints[constraint_name], \
                "Number of parameters exceeds model size constraints!"
        else:
            if use_test_data:
                logging.info("Constraints are checked on a separate test set")
            else:
                logging.info("Constraints are checked on a cross validation sets ")

            if (constraint_name == "n_params"):
                logging.info(f"The constraint {constraint_name}: "
                             f"{optimized_metrics[constraint_name]} <= {constraints[constraint_name]} is satisfied? "
                             f"{optimized_metrics[constraint_name] <= constraints[constraint_name]}")
                if (optimized_metrics[constraint_name] > constraints[constraint_name]):
                    check = False
            else:
                logging.info(f"The constraint {constraint_name}: "
                             f"{optimized_metrics[constraint_name]} >= {constraints[constraint_name]} is satisfied? "
                             f"{optimized_metrics[constraint_name] >= constraints[constraint_name]}")
                if (optimized_metrics[constraint_name] < constraints[constraint_name]):
                    check = False

    print('Resulting Model Score:')
    print(' acc [%]')
    print(optimized_metrics['top3_accuracy'])
    if (optimized_metrics['top3_accuracy'] > min_acc and check == True and use_test_data):
        with open(save_model_str + 'final_model_test.json', 'w') as f:
            json.dump(model_config, f)
            f.write("\n")
    if (optimized_metrics['top3_accuracy'] > min_acc and check == True and use_test_data == False):
        with open(save_model_str + 'final_model_cv.json', 'w') as f:
            json.dump(model_config, f)
            f.write("\n")

    if use_test_data:
        with open(save_model_str + 'result_test.json', 'a+') as f:
            json.dump(optimized_metrics, f)
            f.write("\n")
    else:
        with open(save_model_str + 'result_cv.json', 'a+') as f:
            json.dump(optimized_metrics, f)
            f.write("\n")


def data_aug_list(cfg, size):
    random.seed(42)
    aug_list = []
    print(cfg)
    if cfg['resize'] == True:
        aug_list.append(transforms.Resize(size))
        print("addresize")
    if cfg['horizontal_flip'] == True:
        print("add flip")
        aug_list.append(transforms.RandomHorizontalFlip(p=cfg['horizontal_flip_prob']))
    if cfg['random_crop'] == True:
        print("add crop")
        aug_list.append(transforms.RandomCrop(size, pad_if_needed=True))
    if cfg['rotate'] == True:
        print("add rotate")
        aug_list.append(transforms.RandomRotation(degrees=(-180, 180)))
    return aug_list


def get_optimizer_and_crit(cfg, model_param, lr, cfg_optimizer):
    if cfg_optimizer == 'SGD':
        model_optimizer = torch.optim.SGD
        optimizer = model_optimizer(model_param, lr=lr, momentum=cfg['momentum'])
    elif cfg_optimizer == 'Adam':
        model_optimizer = torch.optim.Adam
        optimizer = model_optimizer(model_param, lr=lr, weight_decay=cfg['weight_decay'])
    else:
        model_optimizer = torch.optim.AdamW
        optimizer = model_optimizer(model_param, lr=lr, weight_decay=cfg['weight_decay'])

    return optimizer


if __name__ == '__main__':
    """
    This is just an example of how you can use train and evaluate
    to interact with the configurable network.

    Also this contains the default configuration you should always capture with your
    configuraiton space!
    """

    cmdline_parser = argparse.ArgumentParser('AutoML SS21 final project')

    cmdline_parser.add_argument('--test_data', default=False, action='store_true',
                                help='use a separate test sets instead of cross validation sets')
    cmdline_parser.add_argument('-e', '--epochs',
                                default=50,
                                help='Number of epochs',
                                type=int)
    cmdline_parser.add_argument('--seed', type=int, default=2, metavar='S',
                                help='random seed (default: 123)')
    cmdline_parser.add_argument('-m', '--model_path',
                                default='/content/drive/MyDrive/run/',
                                help='Path to store model',
                                type=str)
    cmdline_parser.add_argument('-f', '--opt_cfg_path',
                                default='/content/drive/MyDrive/run/_7_2_s_True/opt_cfg.json',
                                help='Path to store model',
                                type=str)

    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    cmdline_parser.add_argument('-s', '--constraint_max_model_size',
                                default=2e7,
                                help="maximal model size constraint",
                                type=int)
    cmdline_parser.add_argument('-p', '--constraint_min_precision',
                                default=0.39,
                                help='minimal constraint constraint',
                                type=float)
    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    # if unknowns:
    #     logging.warning('Found unknown arguments!')
    #     logging.warning(str(unknowns))
    #     logging.warning('These will be ignored')
    # in practice, this could be replaced with "optimal_configuration"
    data = []
    with open(args.opt_cfg_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print(args.test_data)

    # architecture parametrization
    # here is only an example about all the possible hyperparameters for initialization
    architecture = {
        'n_conv_layers': 3,
        'n_channels_conv_0': 457,
        'n_channels_conv_1': 511,
        'n_channels_conv_2': 38,
        'kernel_size': 5,
        'global_avg_pooling': True,
        'batch_norm': False,
        'n_fc_layers': 2,
        'n_channels_fc_0': 27,
        'n_channels_fc_1': 17,
        'n_channels_fc_2': 273,
        'dropout_rate': 0.2}
    for opt_cfg in data:
        print(opt_cfg)
        opt_cfg = opt_cfg.get("configuration")
        print("opt", opt_cfg)
        main(
            opt_cfg,
            # data_dir=optimal_cfg.get("data_dir", os.path.join(os.path.dirname(os.path.abspath(__file__)),
            #                                                  '..', 'micro17flower')),
            data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'micro17flower'),
            use_test_data=args.test_data,
            num_epochs=args.epochs,
            batch_size=opt_cfg.get("batch_size", 282),
            learning_rate=opt_cfg.get("learning_rate_init", 2.244958736283895e-05),
            # train_criterion=loss_dict[opt_cfg.get("training_loss", "cross_entropy")],
            model_optimizer=opt_cfg.get("optimizer", "AdamW"),
            data_augmentations=None,  # Not set in this example
            constraints=OrderedDict(
                [('n_params', args.constraint_max_model_size), ('precision', args.constraint_min_precision)]),
            save_model_str=args.model_path,
        )
