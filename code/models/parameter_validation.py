import fan_tclab_gym as ftg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

folder_path_txt = "../hidden/box_folder_path.txt"
with open(folder_path_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]
box_folder_path = content[0]
# file_path = "/data/heater_0_100_fan_0.4_0.4.csv"
# file_path = "/data/heater_100_100_fan_0.2_1.0.csv"
file_path = "/data/real_perfect_test_step(9).csv"
df = pd.read_csv(box_folder_path + file_path)

start = 6000
stop = start + 6001
start = 0
stop = len(df)
d_traj = df.fan_pwm[start:stop] * 100
d_traj = d_traj.values
h_traj = df.heater_pwm[start:stop]
h_traj = h_traj.values
init_temp = df.temp[start]
# amb_temp = df.amb_temp[0]
amb_temp = 24.18

# c1 = 0.00088341
c2 = 0.801088
# c3 = 0.0039
# c4 = 0.1

# c1 = 0.00075228
# c3 = 0.00358616
c1 = 0.001
c3 = 0.00381
c4 = 0.0899595
c1 = 0.00066225
c3 = 0.00273452

model = ftg.FanTempControlLabBlackBox(initial_temp=init_temp,
                                      amb_temp=amb_temp,
                                      dt=1,
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
t = df.time[0:len(states)]
fig, ax = plt.subplots(3, figsize=(10, 7))
ax[0].plot(t, actions, 'b--', linewidth=3)

ax[0].set_ylabel('PWM %')
ax[0].legend(['Heater'], loc='best')

ax[1].plot(t, df.temp.values[start:stop],
           'bo', linewidth=3, label=r'$T_{c,m}$')
ax[1].plot(t, states[:, 0], 'b-', linewidth=3, label=r'$T_c$')
ax[1].plot(t, states[:, 1], 'r--', linewidth=3, label=r'$T_h$')
ax[1].set_ylabel(r'Temperature (K)')
ax[1].legend(loc='best')

ax[2].plot(t, dists, 'b-', linewidth=3, label=r'Fan',
           alpha=0.5)
ax[2].plot(t, d_traj, 'b-', linewidth=3, label=r'Fan',
           alpha=0.5)
ax[2].set_ylabel('PWM %')
ax[2].set_xlabel('Time (min)')
ax[2].legend(loc='best')
plt.show()
#
