import fan_tclab_gym as ftg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds

folder_path_txt = "../hidden/box_folder_path.txt"
with open(folder_path_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]
box_folder_path = content[0]
file_path = "/data/heater_0_100_fan_0.4_0.4.csv"
df = pd.read_csv(box_folder_path + file_path)

c1s = []
c2s = []
c3s = []
c4s = []
objs = []
heater_pwms = []
fan_pwms = []

d_traj = df.fan_pwm.values * 100
h_traj = df.heater_pwm.values
init_temp = df.temp[0]
dt = np.mean(df.time[0 + 1:len(df)].values
             - df.time[0:len(df) - 1].values)
max_time = len(df) - 0
temp_data = df.temp.values
amb_temp = df.amb_temp[0]

ic1 = 0.00088341
ic2 = 0.801088
ic3 = 0.00388592
ic4 = 1
init_cs = [ic1, ic3]


def sim_model(cs):
    c1, c3 = cs
    c2 = ic2
    c4 = ic4
    model = ftg.FanTempControlLabBlackBox(initial_temp=init_temp,
                                          amb_temp=amb_temp,
                                          dt=dt,
                                          max_time=max_time - 1,
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
    return states[:, 0]


def objective(cs):
    # simulate model
    ym = sim_model(cs)
    # calculate objective
    obj = 0.0
    obj = np.sum((ym - temp_data)**2)
        # return result
    return obj


cs0 = np.zeros(2)
for i in range(len(cs0)):
    cs0[i] = init_cs[i]
    # cs0[2] = 1
# else:
#     cs0[0] = c1s[-1]
#     cs0[1] = c3s[-1]
# cs0[2] = c4s[-1]
# cs0[1] = 0.6
print('Initial SSE Objective: ' + str(objective(cs0)))
bnds = ((0, 1e20), (0, 1e20))
solution = minimize(objective, cs0, method='L-BFGS-B', bounds=bnds)
c_sols = solution.x

print('Final SSE Objective: ' + str(objective(c_sols)))

c1s.append(c_sols[0])
c2s.append(ic2)
c3s.append(c_sols[1])
c4s.append(ic4)
objs.append(objective(c_sols))

heater_pwms.append(h_traj[0])
fan_pwms.append(d_traj[0])
# data_dict = {'heater_pwm': heater_pwms,
#              'fan_pwm': fan_pwms,
#              'c1': c1s,
#              'c2': c2s,
#              'c3': c3s,
#              'c4': c4s,
#              'obj': objs}
# df = pd.DataFrame(data_dict)
# df.to_csv('heater_0_100_fan_0.2_0.2_parameter_results.csv')
# t = df.time[0:len(states)]
# fig, ax = plt.subplots(3, figsize=(10, 7))
# ax[0].plot(t, actions, 'b--', linewidth=3)
#
# ax[0].set_ylabel('PWM %')
# ax[0].legend(['Heater'], loc='best')
#
# ax[1].plot(df.time.values, df.temp.values,
#            'bo', linewidth=3, label=r'$T_{c,m}$')
# ax[1].plot(t, states[:, 0], 'b-', linewidth=3, label=r'$T_c$')
# ax[1].plot(t, states[:, 1], 'r--', linewidth=3, label=r'$T_h$')
# ax[1].set_ylabel(r'Temperature (K)')
# ax[1].legend(loc='best')
#
# ax[2].plot(t, dists, 'b-', linewidth=3, label=r'Fan',
#            alpha=0.5)
# ax[2].plot(t, d_traj, 'b-', linewidth=3, label=r'Fan',
#            alpha=0.5)
# ax[2].set_ylabel('PWM %')
# ax[2].set_xlabel('Time (min)')
# ax[2].legend(loc='best')
# plt.show()
#
