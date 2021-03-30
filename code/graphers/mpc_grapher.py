import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.proportion as prop
import matplotlib
import scipy.stats

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 30}
matplotlib.rc('font', **font)
plt.close('all')



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


tests = ['perfect', 'nominal']
for test in tests:
    folder_path_txt = "../hidden/box_folder_path.txt"
    with open(folder_path_txt) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    box_folder_path = content[0]
    pid_info_file_path = "/data/pid_heat_info.csv"
    pid_info_df = pd.read_csv(box_folder_path + pid_info_file_path)
    savings_array = []
    fails = 0
    n_steps = 0
    dqs = []
    for i in range(1, 4):
        case = i
        # test = 'nominal'
        if test == 'perfect':
            trial_number = 5
        else:
            trial_number = 6

        file_path = "/data/real_{0}_test_case_{1}({2}).csv".format(test,
                                                                   case,
                                                                   trial_number)
        # file_path = "/data/real_nominal_test_step(12).csv"
        folder_path_txt = "../hidden/box_folder_path.txt"
        with open(folder_path_txt) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        box_folder_path = content[0]

        df = pd.read_csv(box_folder_path + file_path)
        tlb = 30
        dt = np.mean(df.time[1:].values
                     - df.time[:- 1].values)
        eval_start = 300
        eval_stop = 5553
        eval_df = df[(df['time'] > eval_start)
                     & (df['time'] < eval_stop)].reset_index(drop=True)
        # total_heat = np.sum(
        #     eval_df.heater_pwm[0:-1].values**2
        #     * (eval_df.time[1:].values - eval_df.time[0:-1].values))
        # pid_heat = [14893318.30305725, 11617602.347231928, 12682570.589524768]
        total_heat = np.sum(
            eval_df.heater_pwm[0:-1].values
            * (eval_df.time[1:].values - eval_df.time[0:-1].values))
        pid_heat = [269914.0478789258, 238483.87268499378, 245240.43045255187]
        max_heat = np.sum(
            np.ones(len(eval_df.heater_pwm[0:-1].values)) * 100
            * (eval_df.time[1:].values - eval_df.time[0:-1].values))
        savings = (1 - total_heat / pid_heat[case - 1]) * 100
        # print(savings)
        fails += len(eval_df[eval_df['temp'] < tlb].index.values)
        n_steps += len(eval_df)
        dqs.extend(list(np.abs(eval_df.heater_pwm[0:-1].values
                               - eval_df.heater_pwm[1:].values)))

        # tlb = df['temp_lb'][0]
        # mini_df = df
        mini_df = df[(df['time'] > 300) & (df['time'] < 5553)]
        # print(mini_df['time'].max())
        # fig, ax = plt.subplots(4)
        # ax[0].plot(mini_df.time,
        #            mini_df.temp, 'bo', label='Measured', markersize=2)
        # ax[0].plot(mini_df.time,
        #            mini_df.est_temp, 'ro', label='Predicted', markersize=2)
        # ax[0].axhline(tlb, color='b', label='$T_{lb}$')
        # ax[0].legend()
        # ax[1].plot(mini_df.time,
        #            mini_df.fan_pwm, 'b-', label='Fan PWM')
        # ax[2].plot(mini_df.time,
        #            mini_df.heater_pwm, 'r-', label='Heater PWM')
        # ax[3].plot(mini_df.time, mini_df.c1, 'g-', label='$c_1$')
        # # ax[3].plot(mini_df.time, mini_df.c2, 'r-', label='$c_2$')
        # ax[3].plot(mini_df.time, mini_df.c3, 'b-', label='$c_3$')
        # # ax[3].plot(mini_df.time, mini_df.c4, 'k-', label='$c_4$')
        # ax[3].legend()
        # plt.show()
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
        # savings = (1 - mini_power / pid_heat_array) * 100
        savings_array.append(savings)

        # fig, ax = plt.subplots(3, sharex=True, figsize=(19, 10))
        # ax[0].plot(time,
        #            mini_df.temp, 'bo-', label='Measured', markersize=4,
        #            linewidth=2)
        # ax[0].plot(time,
        #            mini_df.est_temp, 'ro-', label='Predicted', markersize=4,
        #            linewidth=2)
        # ax[0].plot(time, np.ones(len(time)) * tlb,
        #            'b--', label='$T_{lb}$', linewidth=6)
        # # ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 2),
        # #              fancybox=True, shadow=True, ncol=3)
        # ax[0].set_ylabel(r'$T_c$ (°C)')
        # ax[1].plot(time,
        #            mini_df.fan_pwm * 100, 'b-', label='Fan PWM', linewidth=2)
        # ax[1].set_ylabel('Fan PWM %')
        # ax[2].plot(time,
        #            mini_df.heater_pwm, 'r-', label='Heater PWM', linewidth=2)
        # ax[2].set_ylabel('Heater PWM %')
        # ax[2].set_xlabel('Time (s)')
        #
        # plt.tight_layout()
        # plt.show()
        # plt.savefig('{0}_mpc_case{1}_cut(1).png'.format(test, case))
        # # plt.savefig('{0}_mpc_case{1}_cut_w_legend(1).png'.format(test, case))
        # # fig.savefig('{0}_mpc_case{1}_cut.eps'.format(test, case),
        # #             format='eps')
        # plt.close('all')
    savings_array = np.array(savings_array)
    mean, lb, ub = mean_confidence_interval(savings_array)
    print("Savings compared to PID")
    print("{0:.4f}±{1:.4f}".format(mean, mean - lb))
    dqs = np.array(dqs)
    mean1, lb1, ub1 = mean_confidence_interval(dqs)
    print("Delta Q")
    print("{0:.4f}±{1:.4f}".format(mean1, mean1 - lb1))
    ci = prop.proportion_confint(fails, n_steps, alpha=0.05, method='wilson')
    print('ci --> %.5f' % ci[0], '  %.5f' % ci[1])
