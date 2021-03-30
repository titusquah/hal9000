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
file_path = "/data/test4_big_fan.csv"
df = pd.read_csv(box_folder_path + file_path)

starts = [0]
stops = []
for i in range(len(df) - 1):
    if (df.fan_pwm.values[i] != df.fan_pwm.values[i + 1]
            and df.heater_pwm.values[i] != df.heater_pwm.values[i + 1]):
        starts.append(i + 1)
        stops.append(i)
stops.append(len(df))
amb_temp = df.temp.values[0]
c1s = []
c2s = []
c3s = []
c4s = []
objs = []
heater_pwms = []
fan_pwms = []
for start, stop in zip(starts, stops):
    d_traj = df.fan_pwm[start:stop].values
    h_traj = df.heater_pwm[start:stop].values
    init_temp = df.temp[start]
    dt = np.mean(df.time[start + 1:stop].values
                 - df.time[start:stop - 1].values)
    max_time = stop - start
    temp_data = df.temp[start:stop].values


    def sim_model(cs):
        c1, c3, c4 = cs
        c2 = 0.6
        model = ftg.FanTempControlLabBlackBox(initial_temp=init_temp,
                                              amb_temp=amb_temp,
                                              dt=dt,
                                              max_time=max_time-1,
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
        for ind1 in range(len(ym)):
            obj = obj + (ym[ind1] - temp_data[ind1]) ** 2
            # return result
        return obj


    cs0 = np.zeros(3)
    if start ==0:
        cs0[0] = 0.1
        cs0[1] = 0.1
        cs0[2] = 0.05
    else:
        cs0[0] = c1s[-1]
        cs0[1] = c3s[-1]
        cs0[2] = c4s[-1]
    # cs0[1] = 0.6
    print('Initial SSE Objective: ' + str(objective(cs0)))
    bnds = ((0, 1e20), (0, 1e20), (0, 1e20))
    solution = minimize(objective, cs0, method='L-BFGS-B', bounds=bnds)
    c_sols = solution.x

    print('Final SSE Objective: ' + str(objective(c_sols)))

    c1s.append(c_sols[0])
    c2s.append(0.6)
    c3s.append(c_sols[1])
    c4s.append(c_sols[2])
    objs.append(objective(c_sols))

    heater_pwms.append(h_traj[0])
    fan_pwms.append(d_traj[0])
data_dict = {'heater_pwm': heater_pwms,
             'fan_pwm': fan_pwms,
             'c1': c1s,
             'c2': c2s,
             'c3': c3s,
             'c4': c4s,
             'obj': objs}
df = pd.DataFrame(data_dict)
df.to_csv('test4_big_fan_parameter_results.csv')
# t = df.time[0:len(states)]
# fig, ax = plt.subplots(3, figsize=(10, 7))
# ax[0].plot(t, actions, 'b--', linewidth=3)
#
# ax[0].set_ylabel('PWM %')
# ax[0].legend(['Heater'], loc='best')
#
# ax[1].plot(df.time.values[start:stop], df.temp.values[start:stop],
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
