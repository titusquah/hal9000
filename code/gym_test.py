from fan_tclab_gym import FanTempControlLabBlackBox as bb_process
from fan_tclab_gym import FanTempControlLabGrayBox as gb_process
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folder_path_txt = "hidden/box_folder_path.txt"
with open(folder_path_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]
box_folder_path = content[0]
file_path = "/data/test4_big_fan.csv"
df = pd.read_csv(box_folder_path + file_path)

start = 0
stop = 6001
d_traj = df.fan_pwm[start:stop] * 100

# d_traj = np.ones(len(d_traj)) * 100

h_traj = df.heater_pwm[start:stop]

model = bb_process(initial_temp=296.15,
                   amb_temp=296.15,
                   dt=0.155,
                   max_time=6000,
                   d_traj=d_traj,
                   temp_lb=296.15,
                   c1=0.001,
                   c2=0.6,
                   c3=1e-2,
                   c4=0.05)

actions = [0]
dists = [0]
states = []
state = model.reset()
states.append(state)
done = False
ind1 = 0
while not done:
    state, reward, done, info = model.step([h_traj[ind1]/100])
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

ax[1].plot(t, states[:, 0], 'b-', linewidth=3, label=r'$T_c$')
ax[1].plot(t, states[:, 1], 'r--', linewidth=3, label=r'$T_h$')
ax[1].set_ylabel(r'Temperature (K)')
ax[1].legend(loc='best')

ax[2].plot(t, dists, 'b-', linewidth=3, label=r'Fan',
           alpha=0.5)
ax[2].set_ylabel('PWM %')
ax[2].set_xlabel('Time (min)')
ax[2].legend(loc='best')
plt.show()
#
