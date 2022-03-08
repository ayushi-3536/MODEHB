"""
See the full description of the datasets in here:
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
"""

from os import path, makedirs
import os
import requests
import pandas as pd
import numpy as np
from numpy import save
import sys
from loguru import logger
from sklearn.datasets import load_svmlight_files
from sklearn.model_selection import train_test_split

# project_root = path.join(path.dirname(path.abspath(__file__)), "..")
SHARED_DATA_S3_BUCKET = 'amper-benchmark-repo'

# define dataset types
REGRESSION = 'regression'
BINARY = 'binary'
MULTICLASS = 'multiclass'

logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}

def load_dataset(path, urls):
    #path = os.getcwd().join('..\\baselines\\problems\\adult\\data')
    logger.debug("current working dir path is :{}",)
    logger.debug("path is :{}",path)
    if not os.path.exists(path):
        os.mkdir(path)

    for url in urls:
        data = requests.get(url).content
        filename = os.path.join(path, os.path.basename(url))
        with open(filename, "wb") as file:
            file.write(data)

def download_data_files():
    urls = ["http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"]
    load_dataset('data', urls)

    columns = ["age", "workclass", "fnlwgt", "education", "education-num","marital-status",
               "occupation", "relationship","race", "sex", "capital-gain", "capital-loss",
               "hours-per-week", "country", "salary"]
    train_data = pd.read_csv('data/adult.data', names=columns,
                                        sep=',', na_values='?')
    test_data = pd.read_csv('data/adult.test', names=columns,
                            sep=',', skiprows=1, na_values='?')

    return train_data,test_data



class LibSvmDataset:
    """
    Dataset using the libsvm format

    The dataset will be stored in S3 and cached locally

    It is assumed that the file contains in there are two files,
    with the suffixes .tr and .te are present
    """
    def __init__(self, dataset_name, dataset_type,label_num_classes=None):
        """
        :param dataset_name:
          used for both as an identifier of the dataset and as a name of the datasets which must
          be name.tr for the training set and name.te for the test set
        :param dataset_type:
          one of DATASET_TYPE_REGRESSION, DATASET_TYPE_BINARY and DATASET_TYPE_MULTICLASS
        :param label_num_classes:
          number of classes, required in the case if multiclass classification
        """
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.label_num_classes = label_num_classes
        self.X_tr = None
        self.y_tr = None
        self.X_va = None
        self.y_va = None

    @property
    def name(self):
        return self.dataset_name

    @property
    def type(self):
        return self.dataset_type

    @property
    def num_classes(self):
        return self.label_num_classes

    def get_features_and_labels(self):
        """
        Helper method that reads the dataset and returns the sparse sp.csr_matrix objects
        for test and training set
        """
        raise NotImplementedError("This was deleted for our purposes.")

    def get_training_data(self):
        if self.X_tr is None or self.y_tr is None:
            self.get_features_and_labels()
        return self.X_tr, self.y_tr

    def get_validation_data(self):
        if self.X_va is None or self.y_va is None:
            self.get_features_and_labels()
        return self.X_va, self.y_va

    def get_test_data(self):
        return None, None


class LibSvmAdultDataset(LibSvmDataset):
    """
    Adult dataset using the libsvm format

    The dataset will be stored in S3 and cached locally. It is assumed that the file is adult.csv.
    """
    def __init__(self, dataset_name, dataset_type, label_num_classes=None):
        super().__init__(dataset_name, dataset_type, label_num_classes)

    @property
    def name(self):
        return self.dataset_name

    @property
    def type(self):
        return self.dataset_type

    @property
    def num_classes(self):
        return self.label_num_classes

    def get_features_and_labels(self):
        """
        Helper method that reads the dataset and returns the sparse sp.csr_matrix objects
        for test and training set
        """
        train_data,test_data = download_data_files()
        X , y  = self.process_adult_data(train_data)
        #X_test, Y_test = self.process_adult_data(test_data)
        split_size = 0.3
        # Creation of Train and Test dataset
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=split_size, random_state=22)
        # Creation of Train and validation dataset


        sensitive_feature_name = 'sex'
        feature_names = [key for key in X_tr.keys()]
        self.x_control = [sensitive_feature_name]
        self.feature_names = np.array(feature_names)
        print(self.feature_names)

        # make sure labels are 0, 1
        label0, label1 = list(sorted(set(y_tr)))
        if (label0, label1) != (0, 1):
            y_tr[y_tr == label0] = 0
            y_tr[y_tr == label1] = 1
            y_va[y_va == label0] = 0
            y_va[y_va == label1] = 1

        self.X_tr = np.array(X_tr)
        self.y_tr = np.array(y_tr)
        self.X_va = np.array(X_va)
        self.y_va = np.array(y_va)

        save('data/x_train.npy', self.X_tr)
        save('data/y_train.npy', self.y_tr)
        save('data/x_val.npy', self.X_va)
        save('data/y_val.npy', self.y_va)
        # save('x_test.npy', X_test)
        # save('y_test.npy', Y_test)
        return self.X_tr, self.y_tr, self.X_va, self.y_va


    def get_training_data(self):
        if self.X_tr is None or self.y_tr is None:
            self.get_features_and_labels()
        return self.X_tr, self.y_tr

    def get_validation_data(self):
        if self.X_va is None or self.y_va is None:
            self.get_features_and_labels()
        return self.X_va, self.y_va

    def get_test_data(self):
        return None, None

    def process_adult_data(self, df):

        data = [df]
        df['marital-status'] = df['marital-status'].replace(
            [' Divorced', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed'], 'Single')
        df['marital-status'] = df['marital-status'].replace([' Married-AF-spouse', ' Married-civ-spouse'], 'Couple')
        df['marital-status'] = df['marital-status'].map({'Couple': 0, 'Single': 1})
        rel_map = {' Unmarried': 0, ' Wife': 1, ' Husband': 2, ' Not-in-family': 3, ' Own-child': 4,
                   ' Other-relative': 5}
        race_map = {' White': 0, ' Amer-Indian-Eskimo': 1, ' Asian-Pac-Islander': 2, ' Black': 3, ' Other': 4}
        df['race'] = df['race'].map(race_map)

        def f(x):
            if x['workclass'] == ' Federal-gov' or x['workclass'] == ' Local-gov' or x['workclass'] == ' State-gov':
                return 'govt'
            elif x['workclass'] == ' Private':
                return 'private'
            elif x['workclass'] == ' Self-emp-inc' or x['workclass'] == ' Self-emp-not-inc':
                return 'self_employed'
            else:
                return 'without_pay'

        df['employment_type'] = df.apply(f, axis=1)
        salary_map = {' <=50K': 1, ' >50K': 0}
        df['relationship'] = df['relationship'].map(rel_map)
        employment_map = {'govt': 0, 'private': 1, 'self_employed': 2, 'without_pay': 3}
        df['employment_type'] = df['employment_type'].map(employment_map)
        df.loc[(df['capital-gain'] > 0), 'capital-gain'] = 1
        df.loc[(df['capital-gain'] == 0, 'capital-gain')] = 0
        df.loc[(df['capital-loss'] > 0), 'capital-loss'] = 1
        df.loc[(df['capital-loss'] == 0, 'capital-loss')] = 0
        print(df['salary'])
        df['salary'] = df['salary'].map(salary_map).astype(int)
        df['sex'] = df['sex'].map({' Male': 1, ' Female': 0}).astype(int)
        df['country'] = df['country'].replace(' ?', np.nan)
        df['workclass'] = df['workclass'].replace(' ?', np.nan)
        df['occupation'] = df['occupation'].replace(' ?', np.nan)
        for dataset in data:
            dataset.loc[dataset['country'] != ' United-States', 'country'] = 'Non-US'
            dataset.loc[dataset['country'] == ' United-States', 'country'] = 'US'
        df['country'] = df['country'].map({'US': 1, 'Non-US': 0}).astype(int)
        df.loc[(df['capital-gain'] > 0), 'capital-gain'] = 1
        df.loc[(df['capital-gain'] == 0, 'capital-gain')] = 0
        df.loc[(df['capital-loss'] > 0), 'capital-loss'] = 1
        df.loc[(df['capital-loss'] == 0, 'capital-loss')] = 0
        df.drop(labels=['workclass', 'education', 'occupation'], axis=1, inplace=True)
        X = df.drop(['salary'], axis=1)
        y = df['salary']

        return X,y
        #X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.2, random_state=5)
        #return X_tr, y_tr, X_va, y_va


LibSvmAdultDataset("adult", "binary").get_features_and_labels()
