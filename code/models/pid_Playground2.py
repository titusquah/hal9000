# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:05:18 2021

@author: richs
"""
import fan_tclab_gym as ftg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

start = 0
stop = 6001

# d_traj = np.ones(6001) * 20 # no disturbance
# d_traj = (np.sin(np.linspace(20,100,6001))+1)/2 * 80 + 20


step_20 = np.ones(100) * 20
step_100 = np.ones(100) * 100
hi = np.array([20])
mini_list = [hi]
counter = 1
ind1 = 0
while counter < stop - start :
    if ind1 % 2 == 0:
        mini_list.append(step_20)
    else:
        mini_list.append(step_100)
    ind1 += 1
    counter += 100
d_traj = np.concatenate(mini_list)

# random disturbance fxn
# file_path = "/data/ff_ratio_test_disturbance.csv"
# folder_path_txt = "../hidden/box_folder_path.txt"
# with open(folder_path_txt) as f:
#     content = f.readlines()
# content = [x.strip() for x in content]
# box_folder_path = content[0]
# total_file_path = box_folder_path + file_path
# data = pd.read_csv(total_file_path)
# d_traj = np.array(data['dist'])
# d_traj = d_traj[0:6001]
init_temp = 296
temp_sp = 303.75
dt = 1

##Constants
# Kc = 1
# Ti = 1
Kc = 0.02
Ti = 70
Ratio = 0.004

model = ftg.FanTempControlLabBlackBox(initial_temp=init_temp,
                                      amb_temp=init_temp,
                                      dt=dt,
                                      max_time=6000,
                                      d_traj=d_traj,
                                      temp_lb=296.15,
                                      c1=0.001,
                                      c2=0.801088,
                                      c3=0.00388592,
                                      c4=0.09)
action = 0  # PMW values from 0-1.
actions = [action]
pidActions = [0]
dists = [0]
states = []  # index 0 is sensor temp the following indexs are heating temp
state = model.reset()
states.append(state)
done = False
ind1 = 0
err = []
while not done:
    state, reward, done, info = model.step([action])
    actions.append(action)
    # state, reward, done, info = model.step([0.5])
    # actions.append(0.5)
    dists.append(info['dist'])
    states.append(state)

    ### do some code here###
    err.append(temp_sp - states[-1][0])  # add to the end of the error list
    A = 1 + dt / Ti  # the first term for the PI equation
    B = 1  # the second term for the PI equation

    lastAction = actions[-1]  # get the last value of action
    lastPidAction = pidActions[-1]

    # discrete form of the PI controller
    pidAction = lastPidAction + Kc * ((A * err[-1]) - err[-1])
    ffAction = Ratio * (dists[-1] - 20)

    # keep action between values of 0 and 1 in terms of n*100 = %PWM
    pidAction = np.clip(pidAction, 0, 1)

    pidActions.append(pidAction)
    action = pidAction + ffAction

    action = np.clip(action, 0, 1)

    # action = 0.35
    ind1 += 1
states = np.array(states) - 273.15
t = np.linspace(0, len(states) * dt, len(states))
fig, ax = plt.subplots(3, figsize=(10, 7))
ax[0].plot(t, actions, 'b--', linewidth=3)

ax[0].set_ylabel('PWM %')
ax[0].legend(['Heater'], loc='best')

ax[1].plot(t, states[:, 0], 'b-', linewidth=3, label=r'$T_c$')
# ax[1].plot(t, states[:, 1], 'r--', linewidth=3, label=r'$T_h$')

ax[1].axhline(30, color='b', label='$T_{lb}$')
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
