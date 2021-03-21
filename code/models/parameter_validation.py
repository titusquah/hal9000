import fan_tclab_gym as ftg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 16}
matplotlib.rc('font', **font)
matplotlib.use('Qt5Agg')

folder_path_txt = "../hidden/box_folder_path.txt"
with open(folder_path_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]
box_folder_path = content[0]
# file_path = "/data/heater_0_100_fan_0.4_0.4.csv"
# file_path = "/data/heater_100_100_fan_0.2_1.0.csv"
# file_path = "/data/real_perfect_test_step(9).csv"
file_path = "/data/test4_big_fan.csv"
df = pd.read_csv(box_folder_path + file_path)

start = 6000
stop = start + 6001
# start = 0
# stop = len(df)
d_traj = df.fan_pwm[start:stop] * 100
d_traj = d_traj.values
h_traj = df.heater_pwm[start:stop]
h_traj = h_traj.values
init_temp = df.temp[start]
dt = np.mean(df.time[start + 1:stop].values
             - df.time[start:stop - 1].values)
# amb_temp = df.amb_temp[0]
amb_temp = df.temp.values[0]

# c1 = 0.00088341
# c2 = 0.801088
# c3 = 0.0039
# c4 = 0.1

# c1 = 0.00075228
# c3 = 0.00358616
# c1 = 0.001
# c3 = 0.00381
# c4 = 0.0899595
# c1 = 0.00066225
# c3 = 0.00273452

c1 = 0.00108739
c2 = 0.801088
c3 = 0.00454695
c4 = 0.09

model = ftg.FanTempControlLabBlackBox(initial_temp=init_temp,
                                      amb_temp=amb_temp,
                                      dt=dt,
                                      max_time=stop - start - 1,
                                      d_traj=d_traj,
                                      temp_lb=296.15,
                                      c1=c1,
                                      c2=c2,
                                      c3=c3,
                                      c4=c4)

actions = [0]
dists = [0]
states = []
state = model.reset()
states.append(state)
done = False
ind1 = 0
while not done:
    state, reward, done, info = model.step([h_traj[ind1] / 100])
    actions.append(h_traj[ind1])
    # state, reward, done, info = model.step([0.5])
    # actions.append(0.5)
    dists.append(info['dist'])
    states.append(state)
    ind1 += 1
states = np.array(states)
t = df.time[start:stop]
# fig, ax = plt.subplots(3, figsize=(12, 8))
#
# ax[0].plot(t, df.temp.values[start:stop],
#            'bo', linewidth=3, label=r'Measured $T_{c}$')
# ax[0].plot(t, states[:, 0], 'r-', linewidth=3, label=r'Predicted $T_c$')
# # ax[1].plot(t, states[:, 1], 'r--', linewidth=3, label=r'$T_h$')
# ax[0].set_ylabel(r'Temperature (°C)')
# ax[0].legend(loc='best')
#
# ax[1].plot(t, actions, 'r-', linewidth=3)
#
# ax[1].set_ylabel('Heater PWM %')
# # ax[2].plot(t, dists, 'b-', linewidth=3, label=r'Fan',
# #            alpha=0.5)
# ax[2].plot(t, d_traj, 'b-', linewidth=3, alpha=1)
# ax[2].set_ylabel('Fan PWM %')
# ax[2].set_xlabel('Time (s)')
# ax[2].legend(loc='best')
plt.tight_layout()
plt.show()
ss_resid = np.sum((df.temp.values[start:stop] - states[:, 0]) ** 2)
ss_total = np.sum((df.temp.values[start:stop]
                   - df.temp.values[start:stop].mean()) ** 2)
r_squared = 1 - (ss_resid / ss_total)
print(r_squared)
n = len(t)
k = 2
adjusted_r_squared = 1-(n-1)/(n-(k+1))*(1-r_squared)
print(adjusted_r_squared)
#
