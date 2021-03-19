# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 19:25:17 2021

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


# Import CSV data file
# Column 1 = time (t)
# Column 2 = input (u)
# Column 3 = output (yp)
#################### File Paths
url = r"C:\Users\Tony\Box\hal9000_box_folder\data\step_test_heater_1.csv"   #Heater File
url1= r"C:\Users\Tony\Box\hal9000_box_folder\data\step_test_fan_50_5.csv"   #Disturbance File
#url2 = r"C:\Users\Tony\Box\hal9000_box_folder\data\dist_cases(1).csv"           # Disturbance Case File
#url = r"C:\Users\kervi\Downloads\step_test_fan_50_3.csv"   #Heater File
#url1= r"C:\Users\kervi\Downloads\step_test_fan_50_2.csv"   #Disturbance File


data = pd.read_csv(url)
data1= pd.read_csv(url1)
#data2 = pd.read_csv(url2)

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

# specify number of steps
ns = len(t)
delta_t = t[1]-t[0]
# create linear interpolation of the u data versus time
uf = interp1d(t,u)

# define first-order plus dead-time approximation    
def fopdt(y,t,uf,Km,taum,thetam):
    # arguments
    #  y      = output
    #  t      = time
    #  uf     = input linear function (for time shift)
    #  Km     = model gain
    #  taum   = model time constant
    #  thetam = model time constant
    # time-shift u
    try:
        if (t-thetam) <= 0:
            um = uf(0.0)
        else:
            um = uf(t-thetam)
    except:
        #print('Error with time extrapolation: ' + str(t))
        um = u0
    # calculate derivative
    dydt = (-(y-yp0) + Km * (um-u0))/taum
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
    for i in range(0,ns-1):
        ts = [t[i],t[i+1]]
        y1 = odeint(fopdt,ym[i],ts,args=(uf,Km,taum,thetam))
        ym[i+1] = y1[-1]
    return ym

# define objective
def objective(x):
    # simulate model
    ym = sim_model(x)
    # calculate objective
    obj = 0.0
    for i in range(len(ym)):
        obj = obj + (ym[i]-yp[i])**2    
    # return result
    return obj

# initial guesses
x0 = np.zeros(3)
x0[0] = 3000 # Km
x0[1] = 25000.0 # taum
x0[2] = -0.1 # thetam

# show initial objective
print('Initial SSE Objective: ' + str(objective(x0)))

# optimize Km, taum, thetam
solution = minimize(objective,x0)

# Another way to solve: with bounds on variables
#bnds = ((-100, 500), (-100, 1000.0), (-100.0, 1000.0))
#solution = minimize(objective,x0,bounds=bnds,method='SLSQP')
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
plt.close()
plt.figure()
plt.subplot(2,1,1)
plt.plot(t,yp,'kx-',linewidth=2,label='Process Data')
plt.plot(t,ym1,'b-',linewidth=2,label='Initial Guess')
plt.plot(t,ym2,'r--',linewidth=3,label='Optimized FOPDT')
plt.ylabel('Output')
plt.legend(loc='best')
plt.subplot(2,1,2)
plt.plot(t,u,'bx-',linewidth=2)
plt.plot(t,uf(t),'r--',linewidth=3)
plt.legend(['Measured','Interpolated'],loc='best')
plt.ylabel('Input Data')
plt.show()
###### Determining parameters for process

t = data1['time'].values - data1['time'].values[0]
u = data1['heater_pwm'].values
yp = data1['temp'].values
u0 = u[0]
yp0 = yp[0]

# specify number of steps
ns = len(t)
delta_t = t[1]-t[0]
# create linear interpolation of the u data versus time
uf = interp1d(t,u)

# define first-order plus dead-time approximation    
#def fopdt(y,t,uf,Km,taum,thetam):
#    # arguments
#    #  y      = output
#    #  t      = time
#    #  uf     = input linear function (for time shift)
#    #  Km     = model gain
#    #  taum   = model time constant
#    #  thetam = model time constant
#    # time-shift u
#    try:
#        if (t-thetam) <= 0:
#            um = uf(0.0)
#        else:
#            um = uf(t-thetam)
#    except:
#        #print('Error with time extrapolation: ' + str(t))
#        um = u0
#    # calculate derivative
#    dydt = (-(y-yp0) + Km * (um-u0))/taum
#    return dydt

# simulate FOPDT model with x=[Km,taum,thetam]
#def sim_model(x):
#    # input arguments
#    Km = x[0]
#    taum = x[1]
#    thetam = x[2]
#    # storage for model values
#    ym = np.zeros(ns)  # model
#    # initial condition
#    ym[0] = yp0
#    # loop through time steps    
#    for i in range(0,ns-1):
#        ts = [t[i],t[i+1]]
#        y1 = odeint(fopdt,ym[i],ts,args=(uf,Km,taum,thetam))
#        ym[i+1] = y1[-1]
#    return ym

# define objective
#def objective(x):
#    # simulate model
#    ym = sim_model(x)
#    # calculate objective
#    obj = 0.0
#    for i in range(len(ym)):
#        obj = obj + (ym[i]-yp[i])**2    
#    # return result
#    return obj

# initial guesses
x0 = np.zeros(3)
x0[0] = .0100 # Km
x0[1] = 300.0 # taum
x0[2] = 5.0 # thetam

# show initial objective
print('Initial SSE Objective: ' + str(objective(x0)))

# optimize Km, taum, thetam
#solution = minimize(objective,x0)

# Another way to solve: with bounds on variables
bnds = ((-100000, 100000), (-100000, 100000.0), (-100000.0, 100000.0))
solution = minimize(objective,x0,bounds=bnds,method='L-BFGS-B')
x1 = solution.x

# show final objective
print('Final SSE Objective: ' + str(objective(x1)))

print('Kp: ' + str(x1[0]))
print('taup: ' + str(x1[1]))
print('thetap: ' + str(x1[2]))

# calculate model with updated parameters
ym1 = sim_model(x0)
ym2 = sim_model(x1)
# plot results
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(t,yp,'kx-',linewidth=2,label='Process Data')
plt.plot(t,ym1,'b-',linewidth=2,label='Initial Guess')
plt.plot(t,ym2,'r--',linewidth=3,label='Optimized FOPDT')
plt.ylabel('Output')
plt.legend(loc='best')
plt.subplot(2,1,2)
plt.plot(t,u,'bx-',linewidth=2)
plt.plot(t,uf(t),'r--',linewidth=3)
plt.legend(['Measured','Interpolated'],loc='best')
plt.ylabel('Input Data')
plt.show()

