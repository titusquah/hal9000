import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# for i in range(1, 7):
case = 2

file_path = "/data/real_ff_pid_test_case_{}(2).csv".format(case)
# file_path = "/data/real_pid_test_case_{}(2).csv".format(case)
folder_path_txt = "../hidden/box_folder_path.txt"
with open(folder_path_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]
box_folder_path = content[0]

df = pd.read_csv(box_folder_path + file_path)

tlb = 30

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

fig, ax = plt.subplots(3)
ax[0].plot(df.time,
           df.temp, 'bo', label='Measured', markersize=2)
ax[0].axhline(tlb, color='b', label='$T_{lb}$')
ax[0].legend()
ax[1].plot(df.time,
           df.fan_pwm, 'b-', label='Fan PWM')
ax[2].plot(df.time,
           df.heater_pwm, 'r-', label='Heater PWM')
plt.show()

