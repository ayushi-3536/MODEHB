import json
import os
import glob
import argparse
import pandas as pd
import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
import tqdm
from loguru import logger
import sys

logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}
def load_json_file(path,spath,idx):
    # Opening JSON file
    f = open(path)
    fs = open(spath+'/'+'_cost_'+str(idx)+'.txt','w')
    data = (f.read().split("\n"))
    print(len(data))
    t=0
    for i in data:
         #print(i)
         line = json.loads(i)
         print(line)
         #t+=line['cost']
         fs.write(str(line['n_params'])+' '+str(line['acc'])+' '+str(line['cost'])+'\n')

    # Closing file
    f.close()
    fs.close()
def _hypervolume_evolution_single_dehb(data,cum_time):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hv = Hypervolume(torch.tensor([-0.0,-8.0], device=device))
    result = {
        'hypervolume': [0.0],
        'walltime': [0.0],
        'evaluations': [0.0]
    }

    print("data shape,",data.shape)
    print("time shape",cum_time.shape)
    for i in tqdm(range(1, data.shape[0]), desc='data extraction'):
        print(i)
        print("nds data:{}",data[:i][is_non_dominated(data[:i])])
        print("hypervol:{}",hv.compute(data[:i][is_non_dominated(data[:i])]))
        result['hypervolume'].append(hv.compute(data[:i][is_non_dominated(data[:i])]))
        result['evaluations'].append(i)
    result['walltime'] = np.cumsum(cum_time)
    return pd.DataFrame(result)


parser = argparse.ArgumentParser()
parser.add_argument('--path', default='results_msehvi', type=str, help='Timeout in sec. 0 -> no timeout')
args = parser.parse_args()
path = args.path
extension = 'json'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(result)
for idx,f in enumerate(result):
    df = pd.read_json(path, lines=True)
    data = np.array(-df['acc'])
    logger.info("accuracy data:{}",data)
    data = np.vstack((data, np.array(-df['num_params'])))
    logger.info("data for numparam:{}",data)
    logger.info("shape of the data:{}",data.shape)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(np.ascontiguousarray(np.asarray(data).T), device=device).float()
    logger.info("complete data:{}", data)
    time = np.array(df['cost'])
    print("time:{}", time.shape)
    # print("data:{}",data)
    # ax.legend(loc="lower right")
    data = _hypervolume_evolution_single_dehb(data, time)
    data.to_csv(path + '\\full_' + str(idx) + '.csv')

