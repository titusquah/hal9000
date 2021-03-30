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


scales = [1.25, 1.5, 2.0, 3.0, 5.0]
for scale in scales:
    fails = 0
    n_steps = 0
    savings_array = []
    dqs = []
    for case in range(1, 4):
        file_path = r"/data/real_forecast_scale_" \
                    r"{0}_test_case_{1}.0(2).csv".format(scale, case)
        folder_path_txt = "../hidden/box_folder_path.txt"
        with open(folder_path_txt) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        box_folder_path = content[0]

        df = pd.read_csv(box_folder_path + file_path)
        tlb = 30
        dt = np.mean(df.time[1:].values
                     - df.time[:- 1].values)
        mini_df = df[(df['time'] > 300) & (df['time'] < 5553)]
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
        savings = (1 - total_heat / pid_heat[case - 1]) * 100
        # print(savings)
        savings_array.append(savings)
        fails += len(eval_df[eval_df['temp'] < tlb].index.values)
        n_steps += len(eval_df)
        dqs.extend(list(np.abs(eval_df.heater_pwm[0:-1].values
                               - eval_df.heater_pwm[1:].values)))

        mini_df = df[(df['time'] > 300) & (df['time'] < 5553)]
        time = mini_df.time.values - mini_df.time.values[0]
        # fig, ax = plt.subplots(3, sharex=True, figsize=(19, 10))
        # ax[0].plot(time,
        #            mini_df.temp, 'bo-', label='Measured', markersize=4,
        #            linewidth=2)
        # ax[0].plot(time,
        #            mini_df.est_temp, 'ro-', label='Predicted', markersize=4,
        #            linewidth=2)
        # ax[0].plot(time, np.ones(len(time)) * tlb,
        #            'b--', label='$T_{lb}$', linewidth=6)
        # ax[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1),
        #              fancybox=True, shadow=True, ncol=1)
        # ax[0].set_ylabel(r'$T_c$ (°C)')
        # orig_forecast = mini_df.forecast.values/scale
        # mini_forecast = np.clip(mini_df.forecast.values, 0, 100)
        # ax[1].plot(time,
        #            mini_forecast, 'r-', label='Scaled forecast Fan PWM')
        # ax[1].plot(time,
        #            orig_forecast, 'r--', label='Forecast Fan PWM')
        # ax[1].plot(time,
        #            mini_df.fan_pwm * 100, 'b-',
        #            label='Measured Fan PWM', linewidth=2)
        # ax[1].set_ylabel('Fan PWM %')
        # ax[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1),
        #              fancybox=True, shadow=True, ncol=1)
        # ax[2].plot(time,
        #            mini_df.heater_pwm, 'r-', label='Heater PWM', linewidth=2)
        # ax[2].set_ylabel('Heater PWM %')
        # ax[2].set_xlabel('Time (s)')
        #
        # plt.tight_layout()
        # plt.show()
        # # fig.savefig(r'forecast_mpc_'
        # #             r'scale_{0}_case{1}'
        # #             r'_cut_w_legend.png'.format(scale, case))
        # # fig.savefig(r'forecast_mpc_'
        # #             r'scale_{0}_case{1}_w_legend_cut.eps'.format(scale, case),
        # #             format='eps')
        # plt.close('all')
    savings_array = np.array(savings_array)
    mean, lb, ub = mean_confidence_interval(savings_array)
    print(scale)
    print("Savings compared to PID")
    print("{0:.4f}±{1:.4f}".format(mean, mean - lb))
    dqs = np.array(dqs)
    mean1, lb1, ub1 = mean_confidence_interval(dqs)
    print("Delta Q")
    print("{0:.4f}±{1:.4f}".format(mean1, mean1 - lb1))
    ci = prop.proportion_confint(fails, n_steps, alpha=0.05, method='wilson')

    print('ci --> %.5f' % ci[0], '  %.5f' % ci[1])
    mean2 = np.mean(ci)
    print("{0:.4f}±{1:.4f}".format(mean2*100, (mean2 - ci[0])*100))
    # eval_start = 300
    # eval_df = df[df['time'] > eval_start].reset_index(drop=True)
    # total_heat = np.sum(
    #     eval_df.heater_pwm[0:-1].values
    #     * (eval_df.time[1:].values - eval_df.time[0:-1].values))
    # max_heat = np.sum(
    #     np.ones(len(eval_df.heater_pwm[0:-1].values)) * 100
    #     * (eval_df.time[1:].values - eval_df.time[0:-1].values))
    # pid_heat = [270237.1992528868, 238824.09596286307, 245612.28424969202]
    # savings = (1 - total_heat / pid_heat[case - 1]) * 100
    # print(savings)
    # violate_inds = np.array(eval_df[eval_df['temp'] < tlb].index.values)
    # eval_time = np.concatenate((eval_df.time.values,
    #                             [dt+eval_df.time.values[-1]]))
    # time_violate = eval_time[violate_inds + 1] - eval_time[violate_inds]
    # total_time_violate = np.sum(time_violate)
    # total_time = eval_df['time'].max()-eval_df['time'].min()
    # percent_violated = total_time_violate/total_time
    # print(percent_violated)
    # tlb = df['temp_lb'][0]
    # mini_df = df
    # # mini_df = df[(df['time'] > 3000) & (df['time'] < 4000)]
    # time = mini_df.time.values - mini_df.time.values[0]
    # fig, ax = plt.subplots(4)
    # ax[0].plot(mini_df.time,
    #            mini_df.temp, 'bo', label='Measured', markersize=2)
    # ax[0].plot(mini_df.time,
    #            mini_df.est_temp, 'ro', label='Predicted', markersize=2)
    # ax[0].plot(time, np.ones(len(time)) * tlb,
    #            'b--', label='$T_{lb}$')
    # ax[0].legend()
    # ax[1].plot(mini_df.time,
    #            mini_df.fan_pwm, 'b-', label='Fan PWM')
    # mini_forecast = np.clip(mini_df.forecast.values / 100, 0, 1)
    # ax[1].plot(mini_df.time,
    #            mini_forecast, 'b--', label='Forecast Fan PWM')
    # ax[2].plot(mini_df.time,
    #            mini_df.heater_pwm, 'r-', label='Heater PWM')
# ax[3].plot(mini_df.time, mini_df.c1, 'g-', label='$c_1$')
# # ax[3].plot(mini_df.time, mini_df.c2, 'r-', label='$c_2$')
# ax[3].plot(mini_df.time, mini_df.c3, 'b-', label='$c_3$')
# # ax[3].plot(mini_df.time, mini_df.c4, 'k-', label='$c_4$')
# ax[3].legend()
# plt.show()

# fig, ax = plt.subplots(3, sharex=True, figsize=(14, 10))
# ax[0].plot(mini_df.time,
#            mini_df.temp, 'bo', label='Measured', markersize=2)
# ax[0].plot(mini_df.time,
#            mini_df.est_temp, 'ro', label='Predicted', markersize=2)
# ax[0].plot(mini_df.time, np.ones(len(mini_df.time)) * tlb,
#            color='b', label='$T_{lb}$')
# ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 0.1),
#              fancybox=True, shadow=True, ncol=3)
# ax[0].set_ylabel(r'Temperature (°C)')
# ax[1].plot(mini_df.time,
#            mini_df.fan_pwm * 100, 'b-', label='Fan PWM')
# ax[1].set_ylabel('Heater PWM %')
# ax[2].plot(mini_df.time,
#            mini_df.heater_pwm, 'r-', label='Heater PWM')
# ax[2].set_ylabel('Fan PWM %')
# ax[2].set_xlabel('Time (s)')
#
# plt.tight_layout()
# plt.show()
