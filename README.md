# MO-DEHB: Evolutionary-based Hyperband for Multi-Objective Optimization

### Getting started
```bash
git clone https://github.com/automl/DEHB.git
cd DEHB/
pip install -r requirements.txt

### Pygmo
If you are using `python >=3.9`, you have to install pygmo with `conda`, because it is currently not on pypi. 

conda config --add channels conda-forge
conda config --set channel_priority strict
conda install pygmo==2.16.1
```
### Running MO-DEHB to optimize multiple objectives
To run multi-objective optimization we require 1 extra parameter mo_strategy: we provide MO-optimization 
using Non-dominated sorted(NDS) with crowding distance(NSGA-II) and NDS with eps-net(EPSNET). 
Below example can help you to get started
* [04 - A generic template to use MODEHB for multi-objectives Hyperparameter Optimization](examples/04_mo_pytorch_mnist_hpo.py)

### Tutorials/Example notebooks

* [00 - A generic template to use DEHB for multi-fidelity Hyperparameter Optimization](examples/00_interfacing_DEHB.ipynb)
* [01 - Using DEHB to optimize 4 hyperparameters of a Scikit-learn's Random Forest on a classification dataset](examples/01_Optimizing_RandomForest_using_DEHB.ipynb)
* [02 - Optimizing Scikit-learn's Random Forest without using ConfigSpace to represent the hyperparameter space](examples/02_using DEHB_without_ConfigSpace.ipynb)
* [03 - Hyperparameter Optimization for MNIST in PyTorch](examples/03_pytorch_mnist_hpo.py)
* [04 - A generic template to use MODEHB for multi-objectives Hyperparameter Optimization](examples/04_mo_pytorch_mnist_hpo.py)


To run PyTorch example: (*note additional requirements*) 
```bash
PYTHONPATH=$PWD python examples/04_mo_pytorch_mnist_hpo.py --mo_strategy="NSGA-II --runtime=3600 \
--min_budget 1 --max_budget 3 

```


### DEHB Hyperparameters

*We recommend the default settings*.
The default settings were chosen based on ablation studies over a collection of diverse problems 
and were found to be *generally* useful across all cases tested. 
However, the parameters are still available for tuning to a specific problem.

The Hyperband components:
* *min\_budget*: Needs to be specified for every DEHB instantiation and is used in determining 
the budget spacing for the problem at hand.
* *max\_budget*: Needs to be specified for every DEHB instantiation. Represents the full-budget 
evaluation or the actual black-box setting.
* *eta*: (default=3) Sets the aggressiveness of Hyperband's aggressive early stopping by retaining
1/eta configurations every round
  
The DE components:
* *strategy*: (default=`rand1_bin`) Chooses the mutation and crossover strategies for DE. `rand1` 
represents the *mutation* strategy while `bin` represents the *binomial crossover* strategy. \
  Other mutation strategies include: {`rand2`, `rand2dir`, `best`, `best2`, `currenttobest1`, `randtobest1`}\
  Other crossover strategies include: {`exp`}\
  Mutation and crossover strategies can be combined with a `_` separator, for e.g.: `rand2dir_exp`.
* *mutation_factor*: (default=0.5) A fraction within [0, 1] weighing the difference operation in DE
* *crossover_prob*: (default=0.5) A probability within [0, 1] weighing the traits from a parent or the mutant



