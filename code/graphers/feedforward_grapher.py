import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.proportion as prop
import matplotlib
import scipy.stats

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 24}
matplotlib.rc('font', **font)
fails = 0
n_steps = 0


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


folder_path_txt = "../hidden/box_folder_path.txt"
with open(folder_path_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]
box_folder_path = content[0]
pid_info_file_path = "/data/pid_heat_info.csv"
pid_info_df = pd.read_csv(box_folder_path + pid_info_file_path)
savings_array = []
heat_means = []
heat_sds = []
heat_ses = []
for i in range(1, 4):
    case = i
    test = 'ff_pid'

    # file_path = "/data/real_ff_pid_test_case_{}(2).csv".format(case)
    file_path = "/data/real_{0}_test_case_{1}(2).csv".format(test, case)

    df = pd.read_csv(box_folder_path + file_path)
    dt = np.mean(df.time[1:].values
                 - df.time[:- 1].values)
    tlb = 30

    eval_start = 300
    eval_df = df[df['time'] > eval_start]
    total_heat = np.sum(
        eval_df.heater_pwm[0:-1].values
        * (eval_df.time[1:].values - eval_df.time[0:-1].values))
    heat_mean = total_heat / (eval_df['time'].max() - eval_df['time'].min())
    heat_means.append(heat_mean)
    heat_sd = np.sqrt((np.sum(
        (eval_df.heater_pwm[0:-1].values - heat_mean) ** 2
        * (eval_df.time[1:].values - eval_df.time[0:-1].values))
               / (eval_df['time'].max() - eval_df['time'].min())))
    heat_sds.append(heat_sd)
    heat_ses.append(heat_sd/(len(eval_df)-1))

    max_heat = np.sum(
        np.ones(len(eval_df.heater_pwm[0:-1].values)) * 100
        * (eval_df.time[1:].values - eval_df.time[0:-1].values))
    pid_heat = [270237.1992528868, 238824.095962863, 245612.2842496919]
    savings = (1 - total_heat / pid_heat[case - 1]) * 100
    # print(total_heat)
    print(savings)
    fails += len(eval_df[eval_df['temp'] < tlb].index.values)
    n_steps += len(eval_df)

    mini_df = df[(df['time'] > 300) & (df['time'] < 5553)]

    time = mini_df.time.values - mini_df.time.values[0]
    mini_time = [0]
    mini_power = [0]
    mini_heater_pwms = mini_df['heater_pwm'].values
    for ind1 in range(len(mini_df) - 1):
        if time[ind1] - mini_time[-1] > 60:
            mini_time.append(mini_time[-1] + 1)
            mini_power.append(0)
        mini_power[-1] += mini_heater_pwms[ind1] * (
                time[ind1 + 1] - time[ind1])
    mini_power = np.array(mini_power)
    pid_heat_array = pid_info_df['case{}'.format(i)]
    chck_ind = min(len(mini_power), len(pid_heat_array))
    mini_power = mini_power[:chck_ind]
    pid_heat_array = pid_heat_array[:chck_ind]
    savings = (1 - mini_power / pid_heat_array) * 100
    savings_array.extend(list(savings))
    # print(mini_df['time'].max())
    # fig, ax = plt.subplots(3, figsize=(14,10))
    # ax[0].plot(time,
    #            mini_df.temp, 'bo-', label='Measured', markersize=2)
    # ax[0].plot(time, np.ones(len(time)) * tlb,
    #            'b--', label='$T_{lb}$')
    # ax[0].set_ylabel(r'Temperature (°C)')
    # ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 2),
    #              fancybox=True, shadow=True, ncol=2)
    # ax[1].plot(time,
    #            mini_df.fan_pwm, 'b-', label='Fan PWM', linewidth=3)
    # ax[1].set_ylabel('Fan PWM %')
    # ax[2].plot(time,
    #            mini_df.heater_pwm, 'r-', label='Heater PWM', linewidth=3)
    # ax[2].set_ylabel('Heater PWM %')
    # ax[2].set_xlabel('Time (s)')
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('{0}_case{1}_cut.png'.format(test, case))
savings_array = np.array(savings_array)
mean, lb, ub = mean_confidence_interval(savings_array)
print("{0:.4f}±{1:.4f}".format(mean, mean - lb))
ci = prop.proportion_confint(fails, n_steps, alpha=0.05, method='wilson')
print('ci --> %.5f' % ci[0], '  %.5f' % ci[1])
