import numpy as np
from matplotlib import pyplot as plt
import argparse
import plotly.express as px
import json
import pandas as pd
import numpy as np
from IPython.display import display
def pareto(costs: np.ndarray):
    """
    Find the pareto-optimal points
    :param costs: (n_points, m_cost_values) array
    :return: (n_points, 1) indicator if point is on pareto front or not.
    """
    assert type(costs) == np.ndarray
    assert costs.ndim == 2

    # first assume all points are pareto optimal
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            # determine all points that have a smaller cost
            all_with_lower_costs = np.any(costs < c, axis=1)
            keep_on_front = np.logical_and(all_with_lower_costs, is_pareto)
            is_pareto = keep_on_front
            is_pareto[i] = True  # keep self
    return is_pareto

def plot_3D_pareto(path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cost100 = np.loadtxt(path)
    cost = cost100
    xs = cost[:, 0]
    ys = cost[:, 1]
    zs = cost[:, 2]
    #ax.scatter(xs, ys, zs)
    ax.scatter(xs, ys, zs, c='orange', marker='X')
    ax.set_xlabel('model_size')
    ax.set_ylabel('precision')
    ax.set_zlabel('accuracy')
    ax.view_init(45, -35)
    plt.show()

def plot_all_runs(path):

    cost = np.loadtxt(path)
    cost1 = cost.copy()
    plt.scatter(cost[:, 0], cost[:, 2], color='blue', marker='o', alpha=0.5)
    cost = cost[cost[:, 0] <= 2e7]
    cost = cost[cost[:, 1] <= -0.39]
    cost = cost[cost[:, 2] < 0.2]
    plt.scatter(cost[:, 0], cost[:, 2], color='red', marker='o', alpha=0.5)
    # pareto_front1= cost[front, :]
    # plt.scatter(pareto[:, 0], pareto[:, 2],color='red', marker='o')
    plt.axvline(x=2e7, color='r', linestyle='-')
    plt.axhline(y=0.25, color='r', linestyle='-')
    plt.title('Sampled Configuration')
    plt.xlabel('model_size')
    plt.ylabel('error')

    plt.xscale('log')
    plt.show()

    plt.scatter(cost1[:, 1], cost1[:, 2], color='blue', marker='o', alpha=0.5)
    plt.scatter(cost[:, 1], cost[:, 2], color='red', marker='o', alpha=0.5)
    # pareto_front1= cost[front, :]
    # plt.scatter(pareto1[:, 1], pareto_front1[:, 2],color='red', marker='o')
    plt.axvline(x=-0.39, color='r', linestyle='-')
    plt.axhline(y=0.25, color='r', linestyle='-')
    plt.title('sampled configuration')
    plt.xlabel('precision')
    plt.ylabel('error')

    # plt.xscale('log')
    plt.show()

def plot_parallel_coordinates(path):
    data = pd.read_json(path, lines=True)
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


def input_arguments():
    parser = argparse.ArgumentParser(description='Optimizing MNIST in PyTorch using DEHB.')

    parser.add_argument('--pareto_path', type=str, default="..//pareto_fit_1631372195.362837.txt",
                        help='file that has all the points on pareto')
    parser.add_argument('--all_fitness', type=str, default="..//every_run_cost_1631372195.357772.txt",
                        help='file that has all the points fitness')
    parser.add_argument('--all_configs', type=str, default="..//dehb_run.json",
                        help='file that has all the points')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = input_arguments()
    plot_3D_pareto(args.pareto_path)
    plot_all_runs(args.all_fitness)
    plot_parallel_coordinates(args.all_configs)

