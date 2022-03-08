import numpy as np
from matplotlib import pyplot as plt
import argparse
import plotly.express as px
import json
import pandas as pd
import numpy as np
from IPython.display import display

#This file is used to plot parallel coordinates across various run
data0 = pd.read_json("..//dehb_run_11_6.json", lines=True)
data1 = pd.read_json("..//dehb_run_7_9.json", lines=True)
data2 = pd.read_json("..//dehb_run_9_1.json", lines=True)
data3 = pd.read_json("..//dehb_run_21_1.json", lines=True)
data4 = pd.read_json("..//dehb_run_12_7.json", lines=True)
data = [data0,data1,data2,data3,data4]
print(data[0])
data = data[0]
    #data = data.loc[(data['top3'] >= 0.8) & (data['precision'] >= 0.42) & (data['n_params'] <= 20000000)]
config= data['configuration']
pd.set_option('display.max_rows', None, 'display.max_columns', None)
config = pd.json_normalize(config)# record_path =['configuration'])
    # (config)

df = pd.concat([config['batch_size'],
                    config['dropout_rate'],config['weight_decay'],
                    config['learning_rate_init'],data['precision'],data["top3"]],axis=1)

fig = px.parallel_coordinates(df, color="top3",
                                 color_continuous_scale=px.colors.diverging.Tealrose,
                                 color_continuous_midpoint=0.5)
fig.show()




