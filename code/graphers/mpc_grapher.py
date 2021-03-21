import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# for i in range(1, 7):
test = 'perfect'
case = 2
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
eval_df = df[df['time'] > eval_start]
total_heat = np.sum(
    eval_df.heater_pwm[0:-1].values
    * (eval_df.time[1:].values - eval_df.time[0:-1].values))
max_heat = np.sum(
    np.ones(len(eval_df.heater_pwm[0:-1].values))*100
    * (eval_df.time[1:].values - eval_df.time[0:-1].values))
pid_heat = 238824.09596286307
savings = (1-total_heat/pid_heat)*100
# tlb = df['temp_lb'][0]
mini_df = df
# mini_df = df[(df['time'] > 3000) & (df['time'] < 4000)]
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

fig, ax = plt.subplots(3, sharex=True, figsize=(14, 10))
ax[0].plot(mini_df.time,
           mini_df.temp, 'bo', label='Measured', markersize=2)
ax[0].plot(mini_df.time,
           mini_df.est_temp, 'ro', label='Predicted', markersize=2)
ax[0].plot(mini_df.time, np.ones(len(mini_df.time)) * tlb,
           color='b', label='$T_{lb}$')
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 0.1),
             fancybox=True, shadow=True, ncol=3)
ax[0].set_ylabel(r'Temperature (°C)')
ax[1].plot(mini_df.time,
           mini_df.fan_pwm * 100, 'b-', label='Fan PWM')
ax[1].set_ylabel('Heater PWM %')
ax[2].plot(mini_df.time,
           mini_df.heater_pwm, 'r-', label='Heater PWM')
ax[2].set_ylabel('Fan PWM %')
ax[2].set_xlabel('Time (s)')

plt.tight_layout()
plt.show()
