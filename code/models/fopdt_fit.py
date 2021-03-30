import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 16}
matplotlib.rc('font', **font)

# Import CSV data file
file_path = "/data/heater_100_100_fan_0.2_1.0.csv"

folder_path_txt = "../hidden/box_folder_path.txt"
with open(folder_path_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]
box_folder_path = content[0]

data = pd.read_csv(box_folder_path + file_path)
data = data
t = data['time'].values[327:] - data['time'].values[327]
u = data['fan_pwm'].values[327:]
yp = data['temp'].values[327:]
u0 = u[0]
yp0 = yp[0]

# specify number of steps
ns = len(t)
delta_t = t[1] - t[0]
# create linear interpolation of the u data versus time
uf = interp1d(t, u)


# define first-order plus dead-time approximation
def fopdt(y, t, uf, Km, taum, thetam):
    # arguments
    #  y      = output
    #  t      = time
    #  uf     = input linear function (for time shift)
    #  Km     = model gain
    #  taum   = model time constant
    #  thetam = model time constant
    # time-shift u
    try:
        if (t - thetam) <= 0:
            um = uf(0.0)
        else:
            um = uf(t - thetam)
    except:
        # print('Error with time extrapolation: ' + str(t))
        um = u0
    # calculate derivative
    dydt = (-(y - yp0) + Km * (um - u0)) / taum
    return dydt


# simulate FOPDT model with x=[Km,taum,thetam]
def sim_model(x):
    # input arguments
    Km = x[0]
    taum = x[1]
    thetam = x[2]
    # storage for model values
    ym = np.zeros(ns)  # model
    # initial condition
    ym[0] = yp0
    # loop through time steps
    for i in range(0, ns - 1):
        ts = [t[i], t[i + 1]]
        y1 = odeint(fopdt, ym[i], ts, args=(uf, Km, taum, thetam))
        ym[i + 1] = y1[-1]
    return ym


# define objective
def objective(x):
    # simulate model
    ym = sim_model(x)
    # calculate objective
    obj = 0.0
    for i in range(len(ym)):
        obj = obj + (ym[i] - yp[i]) ** 2
    # return result
    return obj


# initial guesses
x0 = np.zeros(3)
x0[0] = -25.78  # Km
x0[1] = 60.0  # taum
x0[2] = 0.0  # thetam

x = x0
# # show initial objective
print('Initial SSE Objective: ' + str(objective(x0)))
# #
# # optimize Km, taum, thetam
# solution = minimize(objective, x0)
#
# # # Another way to solve: with bounds on variables
bnds = ((-50, 0), (0.01, 100), (0.0, 30.0))
solution = minimize(objective, x0, bounds=bnds, method='L-BFGS-B')
x = solution.x

# show final objective
print('Final SSE Objective: ' + str(objective(x)))

print('Kp: ' + str(x[0]))
print('taup: ' + str(x[1]))
print('thetap: ' + str(x[2]))

# calculate model with updated parameters
ym1 = sim_model(x0)
ym2 = sim_model(x)
# plot results
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(t, yp, 'kx', linewidth=2, label='Process Data')
# plt.plot(t, ym1, 'b-', linewidth=2, label='Initial Guess')
plt.plot(t, ym2, 'r--', linewidth=3, label='Fitted Model')
plt.ylabel('Temperature (°C)')
plt.legend(loc='best')
plt.subplot(2, 1, 2)
plt.plot(t, u, 'bx-', linewidth=2)
# plt.plot(t, uf(t), 'r--', linewidth=3)
# plt.legend(['Measured', 'Interpolated'], loc='best')
plt.ylabel('Heater PWM (%)')
plt.xlabel('Time (s)')
plt.show()
