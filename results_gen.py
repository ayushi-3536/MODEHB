#%matplotlib inline
import numpy as np

from typing import Optional
from matplotlib import pyplot as plt
def plot_perf_over_time(
    ax: plt.Axes,
    results: np.ndarray,
    times: np.ndarray,
    time_step_size: int,
    is_time_cumulated: bool,
    label: str,
    show: bool = True,
    color: str = "red",
    runtime_upper_bound: Optional[float] = None,
) -> None:
    """
    Args:
        results (np.ndarray):
            The performance of each experiment per evaluation.
            The shape must be (n_experiments, n_evals).
        times (np.ndarray):
            The runtime of each evaluation or cumulated runtime over each experiment.
            The shape must be (n_experiments, n_evals).
        time_step_size (int):
            How many time step size you would like to use for the visualization.
        is_time_cumulated (bool):
            Whether the `times` array already cumulated the runtime or not.
        label (str):
            The name of the plot.
        show (bool):
            Whether showing the plot or not.
            If you would like to pile plots, you need to make it False.
        color (str):
            Color of the plot.
        runtime_upper_bound (Optional[float]):
            The upper bound of runtime to show in the visualization.
            If None, we determine this number by the maximum of the data.
            You should specify this number as much as possible.
    """
    n_experiments, n_evals = results.shape
    results = np.maximum.accumulate(results, axis=-1)

    if not is_time_cumulated:
        times = np.cumsum(np.random.random((n_experiments, n_evals)), axis=-1)
    if runtime_upper_bound is None:
        runtime_upper_bound = times[:, -1].max()

    dt = runtime_upper_bound / time_step_size

    perf_by_time_step = np.full((n_experiments, time_step_size), 1.0)
    curs = np.zeros(n_experiments, dtype=np.int32)

    for it in range(time_step_size):
        cur_time = it * dt
        for i in range(n_experiments):
            while curs[i] < n_evals and times[i][curs[i]] <= cur_time:
                curs[i] += 1
            if curs[i]:
                perf_by_time_step[i][it] = results[i][curs[i] - 1]

    T = np.arange(time_step_size) * dt
    mean = perf_by_time_step.mean(axis=0)
    ste = perf_by_time_step.std(axis=0) / np.sqrt(n_experiments)
    ax.plot(T, mean, color=color, label=label)
    ax.fill_between(T, mean - ste, mean + ste, color=color, alpha=0.2)
    ax.set_xscale("log")
    #ax.set_yscale("log")
    ax.grid()
    ax.legend(loc="lower right")

############plot ##################
cost = np.loadtxt("C:\\Users\\ayush\\PycharmProjects\\MODEHB\\MODEHB\\modehb_10_runs\\fashion_logs_18\\pareto_fit_1637268608.0068002.txt")

cost1 = np.loadtxt("C:\\Users\\ayush\\PycharmProjects\\MODEHB\\MODEHB\\modehb_10_runs\\fashion_logs_15\\pareto_fit_1637316280.620008.txt")
cost2 = np.loadtxt("C:\\Users\\ayush\\PycharmProjects\\MODEHB\\MODEHB\\modehb_10_runs\\fashion_logs_14\\pareto_fit_1637315855.4770465.txt")

plt.scatter(10**cost[:, 0],-cost[:, 1], color='green', marker='o')
plt.scatter(10**cost1[:, 0],-cost1[:, 1],color='blue', marker='x')
plt.scatter(10**cost2[:, 0],-cost2[:, 1],color='red', marker='.')

plt.xlim(10**2, 10**8)
plt.ylim(100,0)
plt.title('MODEHB Pareto Front: Fashion dataset')
plt.ylabel('validation-acc')
plt.xlabel('model_param')
plt.legend(loc="upper right")
plt.xscale('log')
plt.show()



############3plot all runs ######################
# cost = np.loadtxt("/content/MODEHB/fashion_runs/fashion_logs_24h_7/every_run_cost_1636267432.795432.txt")
# cost1 = np.loadtxt("/content/MODEHB/fashion_runs/fashion_logs_24h_8/every_run_cost_1636267072.549627.txt")
# cost2 = np.loadtxt("/content/MODEHB/fashion_runs/fashion_logs_24h_10/every_run_cost_1636364329.0508657.txt")
#
# plt.scatter(10**cost[:, 0],-cost[:, 1], color='green', marker='o',label="24h,seed=7",alpha=0.5)
# plt.scatter(10**cost1[:, 0],-cost1[:, 1],color='blue', marker='x',label="24h,seed=8",alpha=0.5)
# plt.scatter(10**cost2[:, 0],-cost2[:, 1],color='red', marker='.',label="24h,seed=10",alpha=0.3)
#
# plt.xlim(10**2, 10**8)
# plt.ylim(100,0)
# plt.title('MODEHB all Sampled Configuration: Fashion dataset')
# plt.ylabel('validation-acc')
# plt.xlabel('model_param')
# plt.legend(loc="upper right")
# plt.xscale('log')
# plt.show()


cost = np.loadtxt("C:\\Users\\ayush\\PycharmProjects\\MODEHB\\MODEHB\\modehb_10_runs\\res\\hv_contribution.txt")
plt.plot(cost[:,0]/3600,cost[:,1],color='green', marker='o',alpha=0.5,label='24h, seed=8')
cost1 = np.loadtxt("C:\\Users\\ayush\\PycharmProjects\\MODEHB\\MODEHB\\modehb_10_runs\\res\\hv_contribution_1.txt")
plt.plot(cost1[:,0]/3600,cost1[:,1],color='blue', marker='x',alpha=0.5,label='24h, seed=7')
cost2 = np.loadtxt("C:\\Users\\ayush\\PycharmProjects\\MODEHB\\MODEHB\\modehb_10_runs\\res\\hv_contribution_2.txt")
plt.plot(cost2[:,0]/3600,cost2[:,1],color='red', marker='.',alpha=0.3,label='24h, seed=10')
plt.title('MODEHB HV')
plt.xlabel('Time')
plt.legend(loc="lower right")
plt.ylabel('HyperVolume')

plt.show()
_, ax = plt.subplots()
for idx, (col, hpo) in enumerate(
        zip(["red"], ["MO-DEHB"])
    ):
        # print(results.shape)
        # print(times.shape)
        c1 = np.random.choice(cost[:,0], cost1[:,0].shape)
        c2 = np.random.choice(cost2[:,0], cost1[:,0].shape)
        # print(cost1[:,0].shape)
        # print(c1.shape)
        t1=c1.T
        t2=cost1[:,0].T
        times = np.stack((t1,t2,c2.T), axis=0)
        #print(times.shape)
        r1 = np.random.choice(cost[:,1], cost1[:,1].shape)
        r = np.random.choice(cost2[:,1], cost1[:,1].shape)
        # print(cost1[:,1].shape)
        # print(r1.shape)
        r1=r1.T
        r2=cost1[:,1].T
        results = np.stack((r1,r2,r.T), axis=0)
        # print(results.shape)
        plot_perf_over_time(
            ax,
            results,
            times,
            time_step_size=500,
            is_time_cumulated=False,
            color=col,
            label=hpo,
        )
plt.show()