"""A common simple Neural Network for the experiments
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
#from .libsvm_dataset import LibSvmAdultDataset
from .fairness_metrics import fairness_risk, STATISTICAL_DISPARITY,UNEQUALIZED_ODDS,UNEQUAL_OPPORTUNITY
from sklearn.preprocessing import StandardScaler
import pathlib
import time
from loguru import logger
import sys
import os

logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}







def evaluate_network(config, budget=None):

    ts_start = time.time()


    budget = budget if budget else config['budget']
    logger.debug("budget for evaluation of config:{}",budget)
    logger.debug("config for evaluation:{}",config)

    # dataset = dataset
    # X_train, y_train, X_val, y_val = dataset.get_features_and_labels()
    # X_test, y_test = dataset.get_validation_data()
    # if dataset.type != 'regression' and len(y_train.shape) > 1:
    #     # compatibility with libsvm datasets, that have targets already one hot encoded
    #     y_train, y_val, y_test = (np.argmax(v, axis=-1) for v in (y_train, y_val, y_test))
    classes = [0,1]#np.unique(y_train)

    # feature_names = dataset.feature_names
    # sensitive_feature_names = dataset.x_control
    sensitive_feature = ['sex']#'[key for key in sensitive_feature_names]
    feature_names = ['age','fnlwgt','education-num','marital-status','relationship','race',
                        'sex' ,'capital-gain', 'capital-loss', 'hours-per-week', 'country',
                        'employment_type']#feature_names.tolist()
    path = lambda x: str(pathlib.Path(__file__).parent.absolute().joinpath('data').joinpath(x))

    X_train = np.load(path('x_train.npy'))#.float()
    #x_train = x_train.permute(0, 3, 1, 2)

    y_train = np.load(path('y_train.npy'))#.long()
    #
    # ds_train = torch.utils.data.TensorDataset(x_train, y_train)
    # ds_train = torch.utils.data.DataLoader(ds_train, , shuffle=True)

    # Read val datasets
    X_val = np.load(path('x_val.npy'))#.float()
    logger.debug("xval shape:{}",X_val.shape)
    #x_val = x_val.permute(0, 3, 1, 2)

    y_val = np.load(path('y_val.npy'))#.long()
    sensitive_rows = X_val[:, feature_names.index(sensitive_feature[0])]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    # Create model
    hidden = [config['n_fc_0'], config['n_fc_1'], config['n_fc_2'], config['n_fc_3']][:config['n_fc_l']]
    mlp = MLPClassifier(hidden_layer_sizes=hidden,
                        alpha=config['alpha'],
                        learning_rate_init=config['lr'],
                        beta_1=config['beta_1'],
                        beta_2=config['beta_2'],
                        tol=config['tol'],
                        verbose=False)
    epoch = []
    for e in range(budget):
        ts_epoch = time.time()
        mlp.partial_fit(X_train, y_train, classes)

        start = time.time()
        # train_score = mlp.score(X_train, y_train)
        # test_score = mlp.score(X_test, y_test)
        # mlp.predict(X_train)
        y_pred_train = mlp.predict(X_train)
        y_pred_test = mlp.predict(X_val)
        prediction_time = time.time() - start

        train_score = accuracy_score(y_train, y_pred_train)
        test_score = accuracy_score(y_val, y_pred_test)

        ts_now = time.time()
        # NOTE: Otherwise experiments train to fast
        # time.sleep((ts_now - ts_epoch) * 9)
        eval_time = ts_now - ts_epoch
        elapsed_time = ts_now - ts_start

        # record all fairness metrics regardless of the chosen constraints
        statistical_disparity = fairness_risk(X_val, y_val, sensitive_rows, mlp, STATISTICAL_DISPARITY)
        unequal_opportunity = fairness_risk(X_val, y_val, sensitive_rows, mlp, UNEQUAL_OPPORTUNITY)
        unequalized_odds = fairness_risk(X_val, y_val, sensitive_rows, mlp, UNEQUALIZED_ODDS)

        epoch.append([1-test_score,statistical_disparity])
    f = open("all_adult_eval.txt",'a+')
    np.savetxt(f,epoch)



        # Ensure to return all quantities of interest

        #
    test_error = 1 - (test_score)

    logger.debug("config:{}, test_error:{}, test_score :{}, train score:{}, dsp:{}, deo :{}, dfp :{}",
                 config, test_error,test_score, train_score, statistical_disparity, unequal_opportunity,unequalized_odds)

    return {
        'train_acc': (train_score, 0.0),
        'test_acc': (test_score, 0.0),
        'test_error': (test_error, 0.0),
        'dsp': (statistical_disparity, 0.0),
        'deo': (unequal_opportunity, 0.0),
        'dfp': (unequalized_odds, 0.0)
    }

