import pickle
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.pareto_frontier import plot_pareto_frontier
import pandas as pd
# open a file, where you stored the pickled data
file = open('C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\shemoa_15k\\SHEMOA_15k01637153594.9189947.pickle', 'rb')
# dump information to that file
data = pickle.load(file)
d = data.fetch_data()
df = vars(d)
df = df['_df']
pd.set_option('max_colwidth', None)

print(df)

df = df.drop(columns=['sem'])

df_acc3 = df[(df['metric_name']=='val_acc_3')]
df_acc3.set_index('trial_index')
df_acc = df[(df['metric_name']=='val_acc_1')]
df_acc.set_index('trial_index')
df_np = df = df[(df['metric_name']=='num_params')]
df_np.set_index('trial_index')
#cost = df_acc['mean']
print("acc:",df_acc)
df_np = df_np.drop(columns=['metric_name','arm_name'])
df_acc = df_acc.drop(columns=['metric_name','arm_name'])

df_acc3 = df_acc3.drop(columns=['metric_name','arm_name'])
print("acc:{}",df_acc)
print("df_scc3:{}",df_acc3)
#print(df_np)

bd = pd.merge(df_np, df_acc,on=['trial_index'])
bd = bd.drop(columns=['trial_index'])

bd = bd.to_numpy()
#print(bd)
print(bd.ndim)
cost=bd
front = pareto(cost)
pareto_front= cost[front, :]
print(pareto_front)
p1= pareto_front

bd3 = pd.merge(df_np, df_acc3,on=['trial_index'])
bd3 = bd3.drop(columns=['trial_index'])

bd3 = bd3.to_numpy()
#print(bd)
print("cost:{}",cost)
cost3=bd3
file_name = open('C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\' + 'shemoa_2_every_run_cost.txt', 'w')
                # for line in costs:
np.savetxt(file_name, cost3)
file_name = open('C:\\Users\\ayush\\OneDrive\\Documents\\courses\\automl\\project\\multiobj\\multi-obj-baselines\\' + 'cost_shemoa_2_every_run_cost.txt', 'w')
                # for line in costs:
np.savetxt(file_name, cost)
from matplotlib import pyplot as plt

plt.scatter(cost[:, 1], cost[:, 0],color='green', marker='o',alpha=0.5,label="sampled_config")
plt.scatter(p1[:, 1], p1[:, 0],color='blue', marker='o',label="pareto")

plt.xlabel('validation-acc')
plt.ylabel('model_param')
plt.legend(loc="upper right")

plt.yscale('log')
plt.show()
plt.yscale('log')
plt.scatter(cost3[:, 1], cost3[:, 0],color='green', marker='o',alpha=0.5,label="sampled_config")

plt.xlabel('validation-acc3')
plt.ylabel('model_param')
plt.legend(loc="upper right")
plt.show()

file.close()