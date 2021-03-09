from gekko import GEKKO
import numpy as np
import fan_tclab_gym as ftg
import matplotlib.pyplot as plt

n_steps = 1001
c1 = 0.00464991
c2 = 0.801088
c3 = 0.0251691
c4 = 0.0184281
# c4 = 0.1
temp_lb1 = 311  # K
dt = 1
d_traj = np.ones(n_steps) * 20
d_traj = np.sin(np.linspace(0, 100, n_steps) / 10) * 40 + 60
step_20 = np.ones(100) * 20
step_100 = np.ones(100) * 100
hi = np.array([20])
mini_list = [hi]
counter = 1
ind1 = 0
while counter < n_steps:
    if ind1 % 2 == 0:
        mini_list.append(step_20)
    else:
        mini_list.append(step_100)
    ind1 += 1
    counter += 100
d_traj = np.concatenate(mini_list)

initial_temp = 311.5
amb_temp = 296.15
log_barrier_tau = 0.5
penalty_scale = 1e5
steepness = 10

env = ftg.FanTempControlLabBlackBox(initial_temp=initial_temp,
                                    amb_temp=amb_temp,
                                    dt=dt,
                                    max_time=n_steps - 1,
                                    d_traj=d_traj,
                                    temp_lb=temp_lb1,
                                    c1=c1,
                                    c2=c2,
                                    c3=c3,
                                    c4=c4)
mpc = GEKKO(name='tclab-mpc', remote=False, server='http://127.0.0.1')
mpc.time = np.linspace(0, 50, 51)
c1 = mpc.Param(value=c1)
c2 = mpc.Param(value=c2)
c3 = mpc.Param(value=c3)
c4 = mpc.Param(value=c4)
temp_lb = mpc.Param(value=temp_lb1)
p = np.zeros(len(mpc.time))
p[-1] = 1.0
final = mpc.Param(value=p)

fan_pwm = mpc.FV(value=20)
# fan_pwm.STATUS = 0
fan_pwm.FSTATUS = 1

heater_pwm = mpc.MV(value=100)
heater_pwm.STATUS = 1
heater_pwm.FSTATUS = 0.
# heater_pwm.DMAX = 20
# heater_pwm.DCOST = 0.1
heater_pwm.LOWER = 0
heater_pwm.UPPER = 100

# State variables
temp_heater = mpc.SV(value=initial_temp)

# Measurements for model alignment
temp_sensor = mpc.SV(value=initial_temp, name='tc1')
# temp_sensor.STATUS = 1  # minimize error between simulation and measurement
temp_sensor.FSTATUS = 1.  # receive measurement

h = mpc.Intermediate(c1 * fan_pwm ** (c2 - 1))
mpc.Equation(temp_heater.dt() == -h * temp_heater
             + c3 * heater_pwm
             + c2 * h * (
                     amb_temp - temp_heater) * fan_pwm)
mpc.Equation((temp_sensor.dt() == c4 * temp_heater - c4 * temp_sensor))
# mpc.Equation((temp_sensor >= temp_lb))
mpc.Obj(mpc.integral(heater_pwm) * final + penalty_scale * mpc.log(
    1 + mpc.exp(steepness * (temp_lb - temp_sensor))) / steepness)
# mpc.Obj((temp_lb-temp_sensor)**2)
# mpc.Obj(mpc.integral(heater_pwm)*final)
mpc.options.IMODE = 6
mpc.options.NODES = 2
mpc.options.SOLVER = 3
mpc.options.COLDSTART = 0
mpc.options.AUTO_COLD = 1

action = 0
actions = [action]
dists = [0]
states = []
state = env.reset()
states.append(state)
done = False
ind1 = 0
while not done:
    state, reward, done, info = env.step([action])
    actions.append(action)
    # state, reward, done, info = env.step([0.5])
    # actions.append(0.5)
    dists.append(info['dist'])
    states.append(state)
    # if ind1 % 10 == 0:
    if ind1 % 50 == 0:
        print(ind1)
    temp_sensor.MEAS = state[0]
    print(temp_sensor.VALUE)
    fan_pwm.MEAS = info['dist']
    try:
        mpc.solve(disp=False)
        if mpc.options.APPSTATUS == 1:
            # Retrieve new values
            action = heater_pwm.NEWVAL / 100
            print(heater_pwm.VALUE)
        else:
            action = 1
    except Exception as e:
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
ax[1].axhline(temp_lb1 - 273.15,
              color='b', linestyle='--', linewidth=3, label=r'$T_{lb}$')
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
