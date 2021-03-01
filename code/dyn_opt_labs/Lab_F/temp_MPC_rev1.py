import numpy as np
import time
import matplotlib.pyplot as plt
import random
# get gekko package with:
#   pip install gekko
from gekko import GEKKO
# get tclab package with:
#   pip install tclab
from tclab import TCLab
import json

fname = '{0}-{1}_{2}-{3}-{4}_'.format(time.localtime().tm_hour,
                                      time.localtime().tm_min,
                                      time.localtime().tm_mon,
                                      time.localtime().tm_mday,
                                      time.localtime().tm_year)


# save txt file
def save_txt(t, Q1, Q2, T1, T2):
    data = np.vstack((t, Q1, Q2, T1, T2))  # vertical stack
    data = data.T  # transpose data
    top = 'Time (sec), Heater 1, Heater 2, ' \
          + 'Temperature 1, Temperature 2'
    np.savetxt('{0}MPC_data.txt'.format(fname), data, delimiter=',',
               header=top, comments='')


# Connect to Arduino
a = TCLab()

# Final time
tf = 10  # min
# number of data points (every 3 seconds)
n = tf * 20 + 1

T_surr = 23
maxT1 = 50.
maxT2 = 40.
minT = 23
seed = 250
cyc_switch = 20
sp2_delay = 10

pc_hor = 60
steps = int(pc_hor / 3) + 1

db1 = 1
db2 = 0.5

# Configure heater levels
# Percent Heater (0-100%)
# %%
# Configure heater levels
# Percent Heater (0-100%)
SP1 = np.ones(n) * T_surr
SP2 = np.ones(n) * T_surr + random.random() * 5
# Heater random steps every 60 steps
# Alternate steps by SP1 and SP2
#  with rapid, random changes every 60 cycles
random.seed(seed)
for i in range(n):
    if i % cyc_switch == 0:
        SP1[i:i + cyc_switch] = SP1[i - 1] + random.random() * 15
    if (i + sp2_delay) % cyc_switch == 0:
        SP2[i:i + cyc_switch] = SP2[i - 1] + random.random() * 8

# Record initial temperatures (degC)
T1m = a.T1 * np.ones(n)
T2m = a.T2 * np.ones(n)
# Store MHE values for plots
Q1s = np.zeros(n)
Q2s = np.zeros(n)

# %%
#########################################################
# Initialize Model as Estimator
#########################################################
m = GEKKO(name='tclab-mpc')
# m.server = 'http://127.0.0.1' # if local server is installed

# 120 second time horizon, 40 steps
m.time = np.linspace(0, pc_hor, steps)

# Parameters to Estimate
K1 = m.Param(value=0.970)
K2 = m.Param(value=0.522)
K3 = m.Param(value=0.492)
tau12 = m.Param(169)
tau3 = m.Param(19)

# Measured inputs
Q1 = m.MV(value=0)
Q1.FSTATUS = 0  # not measured
Q1.STATUS = 1
Q1.DCOST = 0.1
Q1.DMAX = 30
Q1.LOWER = 0
Q1.UPPER = 100

Q2 = m.MV(value=0)
Q2.FSTATUS = 0  # not measured
Q2.STATUS = 1
Q2.DCOST = 0.1
Q2.DMAX = 20
Q2.LOWER = 0
Q2.UPPER = 100

# State variables
TH1 = m.SV(value=T1m[0])
TH2 = m.SV(value=T2m[0])

# Measurements for model alignment
TC1 = m.CV(value=T1m[0], name='tc1')
TC1.STATUS = 1  # minimize error between simulation and measurement
TC1.FSTATUS = 1  # receive measurement
TC1.TR_INIT = 1

TC1.TAU = 15

TC2 = m.CV(value=T2m[0], name='tc2')
TC2.STATUS = 1  # minimize error between simulation and measurement
TC2.FSTATUS = 1  # receive measurement
TC2.TR_INIT = 1

TC2.TAU = 10

Ta = m.Param(value=T_surr)  # degC

# Heat transfer between two heaters
DT = m.Intermediate(TH2 - TH1)

# Empirical correlations
m.Equation(tau12 * TH1.dt() + (TH1 - Ta) == K1 * Q1 + K3 * DT)
m.Equation(tau12 * TH2.dt() + (TH2 - Ta) == K2 * Q2 - K3 * DT)
m.Equation(tau3 * TC1.dt() + TC1 == TH1)
m.Equation(tau3 * TC2.dt() + TC2 == TH2)

# Global Options
m.options.IMODE = 6  # MPC
m.options.EV_TYPE = 1  # Objective type
m.options.NODES = 3  # Collocation nodes
m.options.SOLVER = 3  # IPOPT
m.options.COLDSTART = 1  # COLDSTART on first cycle
##################################################################
# Create plot
plt.close('all')
plt.figure(figsize=(10, 7))
plt.ion()
plt.show()

# Main Loop
start_time = time.time()
prev_time = start_time
tm = np.zeros(n)

try:
    for i in range(1, n):
        # Sleep time
        sleep_max = 3.0
        sleep = sleep_max - (time.time() - prev_time)
        if sleep >= 0.01:
            time.sleep(sleep - 0.01)
        else:
            time.sleep(0.01)

        # Record time and change in time
        t = time.time()
        dt = t - prev_time
        prev_time = t
        tm[i] = t - start_time

        # Read temperatures in Celsius 
        T1m[i] = a.T1
        T2m[i] = a.T2

        # Insert measurements
        TC1.MEAS = T1m[i]
        TC2.MEAS = T2m[i]

        TC1.SPHI = SP1[i] + db1
        TC1.SPLO = SP1[i] - db1

        TC2.SPHI = SP2[i] + db2
        TC2.SPLO = SP2[i] - db2

        # Predict Parameters and Temperatures with MHE
        # use remote=False for local solve
        m.solve()  # disp=False)

        if m.options.APPSTATUS == 1:
            # Retrieve new values
            Q1s[i + 1] = Q1.NEWVAL
            Q2s[i + 1] = Q2.NEWVAL
            with open(m.path + '//results.json') as f:
                results = json.load(f)

        else:
            Q1s[i + 1] = 0
            Q2s[i + 1] = 0

        # Write new heater values (0-100)
        a.Q1(Q1s[i])
        a.Q2(Q2s[i])

        # Plot
        plt.clf()
        ax = plt.subplot(3, 1, 1)
        ax.grid()
        plt.plot(tm[0:i + 1], SP1[0:i + 1] + db1, 'k-', label=r'$T_1$ SP')
        plt.plot(tm[0:i + 1], SP1[0:i + 1] - db1, 'k-', label='')
        plt.plot(tm[0:i + 1], T1m[0:i + 1], 'ro', label=r'$T_1$ meas')
        plt.plot(tm[i] + m.time, results['tc1.bcv'], 'r-', label=r'$T_1$ Pred')
        plt.plot(tm[i] + m.time, results['tc1.tr_hi'], 'k--',
                 label=r'$T_1$ Traj')
        plt.plot(tm[i] + m.time, results['tc1.tr_lo'], 'k--', label=r'')
        plt.ylabel('Temperature (degC)')
        plt.legend(loc=2)

        ax = plt.subplot(3, 1, 2)
        ax.grid()
        plt.plot(tm[0:i + 1], SP2[0:i + 1] + db2, 'k-', label=r'$T_2$ SP')
        plt.plot(tm[0:i + 1], SP2[0:i + 1] - db2, 'k-', label='')
        plt.plot(tm[0:i + 1], T2m[0:i + 1], 'bo', label=r'$T_2$ meas')
        plt.plot(tm[i] + m.time, results['tc2.bcv'], 'b-', label=r'$T_2$ Pred')
        plt.plot(tm[i] + m.time, results['tc2.tr_hi'], 'k--',
                 label=r'$T_2$ Traj')
        plt.plot(tm[i] + m.time, results['tc2.tr_lo'], 'k--', label=r'')
        plt.ylabel('Temperature (degC)')
        plt.legend(loc=2)

        ax = plt.subplot(3, 1, 3)
        ax.grid()
        plt.plot([tm[i], tm[i]], [0, 100], 'k-', label=r'Current Time')
        plt.plot(tm[0:i + 1], Q1s[0:i + 1], 'r-', label=r'$Q_1$')
        plt.plot(tm[0:i + 1], Q2s[0:i + 1], 'b-', label=r'$Q_2$')
        plt.plot(tm[i] + m.time, Q1.value, 'r--', label=r'$Q_1$ forecast')
        plt.plot(tm[i] + m.time, Q2.value, 'b--', label=r'$Q_2$ forecast')
        plt.plot(tm[i] + m.time[1], Q1.value[1], 'ro', label=r'')
        plt.plot(tm[i] + m.time[1], Q2.value[1], 'bo-', label=r'')

        plt.ylabel('Heaters')
        plt.xlabel('Time(s)')
        plt.legend(loc='best')

        plt.draw()
        plt.pause(0.05)

    # Turn off heaters
    a.Q1(0)
    a.Q2(0)
    save_txt(tm, Q1s, Q2s, T1m, T2m)
    # Save figure
    plt.savefig('{0}MPC_pic.png'.format(fname))
    a.close()

# Allow user to end loop with Ctrl-C           
except KeyboardInterrupt:
    # Disconnect from Arduino
    a.Q1(0)
    a.Q2(0)
    print('Shutting down')
    a.close()
    plt.savefig('{0}pic.png'.format(fname))
    raise

# Make sure serial connection still closes when there's an error
except:
    # Disconnect from Arduino
    a.Q1(0)
    a.Q2(0)
    print('Error: Shutting down')
    a.close()
    plt.savefig('{0}pic.png'.format(fname))
    raise
