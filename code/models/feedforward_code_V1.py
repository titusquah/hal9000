# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 14:24:56 2021

@author: kervi
"""

import numpy as np
import math 
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import sympy as sym
from sympy.abc import s,t,x,y,z
from sympy.integrals import inverse_laplace_transform
from sympy.integrals import laplace_transform
from scipy.integrate import odeint
import random
from fan_tclab_gym import FanTempControlLabBlackBox as bb_process
from utils import get_d_traj
import time
from tclab import TCLab
import pyfirmata
from tclab_modules import set_initial_temp, fan_cooling



# Import CSV data file
# Column 1 = time (t)
# Column 2 = input (u)
# Column 3 = output (yp)
#################### File Paths
url = r"C:\Users\Tony\Box\hal9000_box_folder\data\step_test_heater_1.csv"   #Heater File
url1= r"C:\Users\Tony\Box\hal9000_box_folder\data\step_test_fan_50_5.csv"   #Disturbance File
url2 = r"C:\Users\Tony\Box\hal9000_box_folder\data\dist_cases(1).csv"           # Disturbance Case File


data = pd.read_csv(url)
data1= pd.read_csv(url1)
data2 = pd.read_csv(url2)

### Collecting Data file paths
#folder_path_txt = "hidden/box_folder_path.txt"
#with open(folder_path_txt) as f:
#    content = f.readlines()
#content = [x.strip() for x in content]
#box_folder_path = content[0]
#file_path = "/data/feedforward_1.csv"


t = data['time'].values - data['time'].values[0]
u = data['fan_pwm'].values
yp = data['temp'].values
u0 = u[0]
yp0 = yp[0]

case=['case1','case2','case3']

filename=[1,2,3]

# specify number of steps
ns = len(t)
delta_t = t[1]-t[0]
# create linear interpolation of the u data versus time
uf = interp1d(t,u)




####### FEEDFORWARD CODE ########

kff=-x[0]/x1[0]
theta_ff=x[2]-x1[2]
lead=x[1]
lag=x1[1]

# Connect to Arduino
#heater_board = TCLab(port='4')
#
##Connect to Fan
#board = pyfirmata.Arduino(
#    "com5")
#
#it = pyfirmata.util.Iterator(board)
#it.start()
#
tf = 1000         # final time
n = int(tf + 1) # number of time points

for i in range (0,len(case)):
	# time span for the simulation, cycle every 1 sec
	ts = np.linspace(0,tf,n)
	delta_t = ts[1] - ts[0]
	
	# disturbances
	DP = np.zeros(n)
	Fout = np.ones(n)*2.0
	Fin = np.zeros(n)
	
	# Desired level (set point)
	SP = 26
	# level initial condition
	Level0 = SP
	
	# initial valve position
	valve = data1['heater_pwm'][0]
	# Controller bias
	ubias = valve
	# valve opening (0-100%)
	u1 = np.ones(n) * valve
	
	Cv = 0.0001     # valve size
	rho = 1000.0 # water density (kg/m^3)
	A = 5.0      # tank area (m^2)
	gs = 1.0     # specific gravity
	
	# for storing the results
	z = np.ones(n)*Level0
	es = np.zeros(n)
	P = np.zeros(n)   # proportional
	I = np.zeros(n)   # integral
	ie = np.zeros(n)
	
	# Controller tuning
	Kc = x1[0]
	tauI = x1[1]
	start=0
	stop=ns
	d_traj = data2[case[i]][start:stop] * 100
	
	h_traj = data1['heater_pwm'][start:stop]
	c1a=0.38890287535165813
	c2a=1.1845783829062957
	c3a=0.26458722910810484
	c4a=0.007265841148016536
	# simulate with ODEINT
	
	folder_path_txt = "../hidden/box_folder_path.txt"
	with open(folder_path_txt) as f:
	    content = f.readlines()
	content = [x.strip() for x in content]
	box_folder_path = content[0]
	suffix=str(filename[i])
	file_path = "/data/feedforward_"+suffix+".csv"
	
	# Connect to Arduino
	heater_board = TCLab(port='4')
	
	#Connect to Fan
	board = pyfirmata.Arduino(
	    "com5")
	
	it = pyfirmata.util.Iterator(board)
	it.start()
	# Main Loop
	start_time = time.time()
	prev_time = start_time
	times = []
	temps = []
	
	sleep_max = 1
	steps_per_second = int(1 / sleep_max)
	
	#heater_pwms = np.concatenate((np.ones(steps_per_second * 5),
	#                              np.ones(steps_per_second * 2400),
	#                              np.ones(steps_per_second * 2400))) * 50
	#fan_pwms=np.concatenate((np.zeros(steps_per_second * 5),
	#                              np.ones(steps_per_second * 2400),
	#                              np.zeros(steps_per_second * 2400))) * 0.5
	
	
	#n = len(data2['case1'])*tf
	d_traj=get_d_traj(0,5)
	pntxt2 = "d:{}:o".format(3)
	dpin = board.get_pin(pntxt2)
	dpin.mode = 3
	j=0
	
	initial_temp = set_initial_temp(dpin, board, SP, 1, 30)
	try:
	    for i in range(1, n):
	        # Sleep time
	        sleep = sleep_max - (time.time() - prev_time)
	        if sleep >= 0.01:
	            time.sleep(sleep - 0.01)
	        else:
	            time.sleep(0.01)
	
	        # Record time and change in time
	        t = time.time()
	        dt = t - prev_time
	        prev_time = t
	        times.append(t - start_time)
	
	        # Read temperatures in Celsius
	        temps.append(heater_board.T1)
	
	        # Write new heater values (0-100)
	        #pntxt2 = "d:{}:o".format(3)
	#        dpin = board.get_pin(fan_pwms[i])
	#        dpin.mode = 3
	        
	        
	        # inlet pressure (bar) disturbance
	        if i&tf==0:
	            
	            DP[i] = data2[case[i]][j]
	            j+=1
	        else:
	            DP[i] = data2[case[i]][j]
	    
	        # inlet mass flow
	        Fin[i] = DP[i]
	    
	        # outlet flow (kg/sec) disturbance (change every 10 seconds)
	        if np.mod(i+1,500)==100:
	            Fout[i] = Fout[i-1] + 10.0
	        elif np.mod(i+1,500)==350:
	            Fout[i] = Fout[i-1] - 10.0
	        else:
	            if i>=1:
	                Fout[i] = Fout[i-1]
	        dpin.write(Fout[i])
	        # PI controller
	        # calculate the error
	        error = SP - Level0
	        P[i] = Kc * error
	        if i >= 1:  # calculate starting on second cycle
	            ie[i] = ie[i-1] + error * delta_t
	            I[i] = (Kc/tauI) * ie[i]
	        valve = ubias + P[i] + I[i] + kff * (Fout[i]-Fout[0])
	        valve = max(0.0,valve)   # lower bound = 0
	        valve = min(100.0,valve) # upper bound = 100
	        if valve > 100.0:  # check upper limit
	            valve = 100.0
	            ie[i] = ie[i] - error * delta_t # anti-reset windup
	        if valve < 0.0:    # check lower limit
	            valve = 0.0
	            ie[i] = ie[i] - error * delta_t # anti-reset windup
	        
	        u1[i] = valve   # store the valve position
	        heater_board.Q1(valve)
	
	        if i % 30 == 0:
	            df = pd.DataFrame({'time': times,
	                               'temp': temps,
	                               'heater_pwm': u1[0:i],
	                                'fan_pwm':DP[0:i]})
	            df.to_csv(box_folder_path + file_path)
	
	    # Turn off heaters
	    heater_board.Q1(0)
	    heater_board.Q2(0)
	    dpin.write(1)
	#    heater_board.close()
	    cool=fan_cooling(dpin,board)
		
	    
	
	
	    df = pd.DataFrame({'time': times,
	                               'temp': temps,
	                               'heater_pwm': u1[0:i],
	                                'fan_pwm':DP[0:i]})
	    df.to_csv(box_folder_path + file_path)
	#plt.figure(3)
	#plt.plot(df['time'],df['temp'],'r--',linewidth=3,label='valve')
	#plt.ylabel('Valve') 
	
	# Allow user to end loop with Ctrl-C
	except KeyboardInterrupt:
	    # Disconnect from Arduino
	    heater_board.Q1(0)
	    heater_board.Q2(0)
	    print('Shutting down')
	    heater_board.close()
	    dpin.write(0)
	    board.exit()
	    df = pd.DataFrame({'time': times,
	                               'temp': temps,
	                               'heater_pwm': u1[0:i],
	                                'fan_pwm':DP[0:i]})
	    df.to_csv(box_folder_path + file_path)
	
	
	# Make sure serial connection still closes when there's an error
	except:
	    # Disconnect from Arduino
	    heater_board.Q1(0)
	    heater_board.Q2(0)
	    print('Error: Shutting down')
	    heater_board.close()
	    dpin.write(0)
	    board.exit()
	    df = pd.DataFrame({'time': times,
	                               'temp': temps,
	                               'heater_pwm': u1[0:i],
	                                'fan_pwm':DP[0:i]})
	    df.to_csv(box_folder_path + file_path)
	    raise

'''

for i in range(n-1):
    # inlet pressure (bar) disturbance
    DP[i] = data2['case1'][i]

    # inlet mass flow
    Fin[i] = DP[i]

    # outlet flow (kg/sec) disturbance (change every 10 seconds)
    if np.mod(i+1,500)==100:
        Fout[i] = Fout[i-1] + 10.0
    elif np.mod(i+1,500)==350:
        Fout[i] = Fout[i-1] - 10.0
    else:
        if i>=1:
            Fout[i] = Fout[i-1]

    # PI controller
    # calculate the error
    error = SP - Level0
    P[i] = Kc * error
    if i >= 1:  # calculate starting on second cycle
        ie[i] = ie[i-1] + error * delta_t
        I[i] = (Kc/tauI) * ie[i]
    valve = ubias + P[i] + I[i] + kff * (Fout[i]-Fout[0])
    valve = max(0.0,valve)   # lower bound = 0
    valve = min(100.0,valve) # upper bound = 100
    if valve > 100.0:  # check upper limit
        valve = 100.0
        ie[i] = ie[i] - error * delta_t # anti-reset windup
    if valve < 0.0:    # check lower limit
        valve = 0.0
        ie[i] = ie[i] - error * delta_t # anti-reset windup

    u[i+1] = valve   # store the valve position
    es[i+1] = error  # store the error
    y = bb_process(initial_temp=297.6, #22,
                       amb_temp=297.6, #22,
                       dt=0.155,
                       max_time=ns-1,
                       d_traj=d_traj,
                       temp_lb=min(data['temp']),#296.15,
                       c1=c1a,
                       c2=c2a,
                       c3=c3a,
                       c4=c4a)
    model=y
    actions = [0]
    dists = [0]
    states = []
    state = model.reset()
    states.append(state)
    done = False
    ind1 = 0
#    state, reward, done, info = model.step([h_traj[ind1]/100])
#    state, reward, done, info = model.step([0.5])
    #return state[:,0]
    
    while not done:
        state, reward, done, info = model.step([h_traj[ind1]/100])
        actions.append(h_traj[ind1])
        # state, reward, done, info = model.step([0.5])
        # actions.append(0.5)
        dists.append(info['dist'])
        states.append(state)
        ind1 += 1
    states = np.array(states)
    Level0 = states[:-1]-273.15 # take the last point
    z[i+1] = Level0 # store the level for plotting
#Fout[n-1] = Fout[n-2]
#DP[n-1] = DP[n-2]
#Fin[n-1] = Fin[n-2]
#ie[n-1] = ie[n-2]
'''
# plot results
plt.figure(1)
#plt.subplot(4,1,1)
#plt.plot(ts,z,'b-',linewidth=3,label='level')
#plt.plot([0,max(ts)],[SP,SP],'k--',linewidth=2,label='set point')
#plt.ylabel('Tank Level')
#plt.legend(loc=1)
#plt.subplot(4,1,2)
plt.plot(df['time'],df['temp'],'r--',linewidth=3,label='valve')
plt.ylabel('Valve')    
#plt.legend(loc=1)
#plt.subplot(4,1,3)
#plt.plot(ts,es,'k:',linewidth=3,label='error')
#plt.plot(ts,ie,'r:',linewidth=3,label='int err')
#plt.legend(loc=1)
#plt.ylabel('Error = SP-PV')    
#plt.subplot(4,1,4)
#plt.plot(ts,Fin,'k-',linewidth=3,label='Inlet Flow (kg/sec)')
#plt.plot(ts,Fout,'b--',linewidth=3,label='Outlet Flow (kg/sec)')
#plt.plot(ts,DP,'r:',linewidth=3,label='Inlet Pressure (bar)')
#plt.ylabel('Disturbances')    
#plt.xlabel('Time (sec)')
#plt.legend(loc=1)
#
#plt.show()



#X = inverse_laplace_transform(((5*s+1)/(4*s+1))*sym.exp(.1*s),s,t)
#print('X')
#print(X)



