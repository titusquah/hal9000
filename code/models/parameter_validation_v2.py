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
# file_path = "/data/step_test_data(5).csv"
file_path = "/data/heater_100_100_fan_0.2_1.0.csv"
# file_path = "/data/heater_0_100_fan_1.0_1.0.csv"
# file_path = "/data/real_perfect_test_step(9).csv"
# file_path = "/data/real_perfect_test_case_4(4).csv"
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
dt = np.mean(df.time[start + 1:stop].values
             - df.time[start:stop - 1].values)
# amb_temp = df.amb_temp[0]
amb_temp = 23.81
guess_cp = 500
guess_alpha = 0.008
guess_tau_hc = 5
guess_kd = 0.1
guess_beta1 = 11
guesses = [guess_cp,
           guess_alpha,
           guess_tau_hc,
           guess_kd,
           guess_beta1]
cs = np.array([4.78993545e+02, 1.27994140e-02, 32, 1.90822949e-01,
       1.07305228e+01])
# cs = np.array([4.79000866e+02, 1.23454967e-02, 3.25009772e+01, 9.88539510e-02,
#                1.06990993e+01])
# cs = np.array([3.68358754e+02, 7.25533973e-03, 2.14053115e+01, 1.63950265e-01,
#  3.59473475e+00])
# cs = np.array([4.79522382e+02, 3.43681778e-03, 3.24255476e+01, 2.42200656e-02,
#  9.53981021e+00])
cp, alpha, tau_hc, kd, beta1 = cs
model = ftg.FanTempControlLabGrayBox(initial_temp=init_temp + 273.15,
                                     amb_temp=amb_temp + 273.15,
                                     cp=cp,
                                     surf_area=1.2e-3,
                                     mass=.004,
                                     emissivity=0.9,
                                     dt=1,
                                     max_time=(stop - 1) - start,
                                     alpha=alpha,
                                     tau_hc=tau_hc,
                                     k_d=kd,
                                     beta1=beta1,
                                     beta2=0.9,
                                     d_traj=d_traj,
                                     temp_lb=296.15,
                                     )

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
states = np.array(states) - 273.15
t = df.time[0:len(states)]
fig, ax = plt.subplots(3, figsize=(10, 7))
ax[0].plot(t, actions, 'b--', linewidth=3)

ax[0].set_ylabel('PWM %')
ax[0].legend(['Heater'], loc='best')

ax[1].plot(t, df.temp.values[start:stop],
           'bo', linewidth=3, label=r'$T_{c,m}$')
ax[1].plot(t, states[:, 0], 'b-', linewidth=3, label=r'$T_c$')
# ax[1].plot(t, states[:, 1], 'r--', linewidth=3, label=r'$T_h$')
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
