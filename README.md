
This repository uses DEHB optimizer which is extended by NSGA-II to optimize model_size, precision and accuracy.
Implementation for DEHB is open-source and is explained in the following research paper:https://arxiv.org/abs/2105.09821
The base implementation is then extended for Multi-Criteria Optimization

@article{awad2021dehb, title={DEHB: Evolutionary Hyberband for Scalable, Robust and Efficient Hyperparameter Optimization},
author={Awad, Noor and Mallik, Neeratyoy and Hutter, Frank}, journal={arXiv preprint arXiv:2105.09821}, year={2021} }


The population is warm-initialized using SMAC for cheap function evaluation, evaluating only model_size.

The population initialization is followed by mutation, recombination and selection of population. The base DE classes such as DE and DEHB
are extended by MODE and MODEHB extending the algorithm for multi criteria optimization.

After completion of AutoML.py(the main optimizer) all configurations that 
satisfy all constraints including accuracy  are saved in opt_cfg.json file which is then loaded in main.py file
and configurations are evaluated on the test set. Model that satisfy constraints
even on test set and has highest top3 accuracy are saved in final_model.json file.
We are also saving configs that just satisfy model size and min precision constraint just fro analysis, prefixed as constraint_satisfied.json
We are also saving pareto and all sampled configuration after every 25 evals, prefixed with pareto_fit_{} and every_run_cost_{}.
These files help us evaluate performance over time.We are also saving each run with configuration and respective metrices in file dehb_run.json.
These files will be generated in output_dir provided by cmd line argument or in the provided default folder

##Summarizing file hierarchy for obtained results:
*Main directory for every run: output_dir_location_{seed}_{run}:(_11_6.zip contains full run for seed:11)

*Optimum configuration obtained by optimizer: opt_cfg.json(the file checked in is for random seed:11)

*Final model chosen by main.py based on highest top-3 accuracy obtained on test-set: main/final_model_test.json

*Final model chosen by main.py based on highest top-3 accuracy obtained by CV folds: main/final_model_cv.json

*Files for every run: dehb_run.json

*Files for pareto cost metrices generating every 25 evaluation: pareto_fit_{time}.txt



## Repo structure
* [micro17flower](micro17flower) <BR>
  contains a downsampled version of a [flower classification dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html).
  We cropped ans resized it such that the resulting images are of size 16x16. You are only allowed to run your optimizer with the data in [train](micro17flower/train)


* [colab_start.ipynb](Google colab script ):
    This file contains all detailed step to do project setup on google colab, run AutoML.py and generate graphs
* [src](src) Source folder      
    * [bohb.py](src/bohb.py) <BR>
      Uses SMAC for cheap function evaluation( model size evaluation) to warmstart the population. Since it is a cheap evaluation and SMAC is sample efficient,
      number of evaluations is set to 25
    
    * [cnn.py](src/cnn.py)<BR>
      contains the source code of the network you need to optimize. It optimizes the top-3 accuracy.
    
    * [main.py](src/main.py)<BR>
      contains an example script that shows you how to instantiate the network and how to evaluate metrics required 
      in this project. This file also gives you the **default configuration** that always has to be in yourserch space.

    * [AutoML.py](src/AutoML.py) <BR>
      Uses DEHB algorithm which is extended by NSGA-II in MoDEHB.py for multi-criteria optimization (network size, precision and accuracy ) and 
      store the optimal configuration to "opt_cfg.json". It also stores configuration which just satisfy the model size constraint and precision 
      constraint but not accuracy constraint in constraint_satisfied_cfg.json
      
    * [modehb.py](src/DEHB/dehb/optimizers/modehb.py) <BR>
      This file extends open-source DEHB algorithm to optimize multiple criteria using NSGA-II. We are optimizing network size, precision and accuracy .
      This file is responsible for the DEHB flow: Initialization call to mode.py to warm start the population, selecting population to mutate using
      NDS and crowding distance, recombination of generated mutant vector with the parent vector, evaluation of the generate configuration,
      selecting the best configuration using NDS and hypervolume configuration.
      This file also maintains top-k candidates from the pareto front to be sampled in case there are not enough samples in the parent pool.
    
    * [mode.py](src/DEHB/dehb/optimizers/mode.py) <BR>
      This file extends base de algorithm. The important function being called from modehb is init_population. This function calls SMAC
      to get configurations satisfying model size constraint.This is done to warm start the population with configuration satisfying mall model size constraints.
      This hierarchy(inheritence) is done to keep the project consistent with the base DEHB repository
      
    * [randomsearch.py](src/randomsearch.py) <BR>
      simple random search implementation to compare performance with DEHB
      
    * [pareto.py](src/pareto_utils/pareto.py) <BR>
      this contains all the pareto specific function such as NDS sorting, crowding distance calculation and identifying pareto
   
    * [utils.py](src/examples/metric_utils.py)<BR>
      contains simple helper functions for cnn.py
      
##Installation guide

####clone the repository
* git clone https://github.com/automl-classroom/automl-ss21-final-project-ayushi-3536.git

####run requirements.txt to install all dependencies
* pip install -r requirements.txt

####run setup.sh
* bash setup.sh

####To run AutoML.py 
* python3 src/AutoML.py --seed 11 --run 1 --runtime 43200 --constraint_min_precision 0.39

####To run main.py 
* python src/main.py -m '/content/drive/MyDrive/run/_7_2_ws_True/' --opt_cfg_path '/content/drive/MyDrive/run/_7_2_ws_True/opt_cfg.json' --constraint_min_precision 0.39 --constraint_max_model_size 20000000




