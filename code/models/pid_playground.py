import fan_tclab_gym as ftg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

start = 0
stop = 6001
d_traj = np.ones(6001) * 20

init_temp = 298
temp_sp = 313
dt = 0.155

model = ftg.FanTempControlLabBlackBox(initial_temp=init_temp,
                                      amb_temp=init_temp,
                                      dt=dt,
                                      max_time=6000,
                                      d_traj=d_traj,
                                      temp_lb=296.15,
                                      c1=0.0007,
                                      c2=0.800573,
                                      c3=0.00395524,
                                      c4=0.00284566)
action = 0
actions = [action]
dists = [0]
states = []
state = model.reset()
states.append(state)
done = False
ind1 = 0
while not done:
    state, reward, done, info = model.step([action])
    actions.append(action)
    # state, reward, done, info = model.step([0.5])
    # actions.append(0.5)
    dists.append(info['dist'])
    states.append(state)

    ### do some code here###
    action = 1
    ind1 += 1
states = np.array(states) - 273.15
t = np.linspace(0, len(states) * dt, len(states))
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
ax[2].plot(t, d_traj, 'b-', linewidth=3, label=r'Fan',
           alpha=0.5)
ax[2].set_ylabel('PWM %')
ax[2].set_xlabel('Time (min)')
ax[2].legend(loc='best')
plt.show()
#
