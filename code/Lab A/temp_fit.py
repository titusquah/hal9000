# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 18:30:01 2019

@author: Titus
"""

import pandas as pd
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

m_data=pd.read_excel("trial4.xlsx")





p=np.asarray(m_data["Q"])
T=np.asarray(m_data["T"])
t=np.asarray(m_data["t"])



dT=np.zeros(len(t)-1)
dt=np.zeros(len(t)-1)
est_temp=np.zeros(len(t)-1)
est_p=np.zeros(len(t)-1)


Q=100 #Percent Heater (0-100%)
T0=T[0]+273.15     #Init temp 
Ta=T[0]+273.15      #Surr Temp
U=10.              #W/m^2 K
mass=4./1000.      #kg
Cp=0.5*1e3         #J/kgK
A=12./100**2       #Area in m^2
alpha =0.01       #W / % heater
eps = 0.9          #Emissivity
sigma =5.67e-8          #Stefan boltzman constant

for i in range(len(dt)):
  dt[i]=np.mean([t[i],t[i+1]])
  dT[i]=(T[i+1]-T[i])/(t[i+1]-t[i])
  est_temp[i]=np.mean([T[i],T[i+1]])
  est_p[i]=np.mean([p[i],p[i+1]])

win=0

def mov_ave(data_list,moving_average_window=5):
  win=moving_average_window
  pv_ave=np.zeros(len(data_list))
  for i in range(len(data_list)):
    if i-win>=0 and i+win<=len(data_list):
        pv_ave[i]=np.mean(data_list[i-win:i+win])
    elif i-win<0:
        pv_ave[i]=np.mean(data_list[i:i+win])
    elif i+win>len(data_list):
        pv_ave[i]=np.mean(data_list[i-win:i])
  return pv_ave
if win!=0:
  dT=mov_ave(dT,win)


def dT_model(X,U):
  T,Q = X
  
  delta_T=(U*A*(Ta-T)+eps*sigma*A*(Ta-T)**4+Q*alpha)/(mass*Cp)
  return delta_T

coeffs, covar=opt.curve_fit(dT_model,(est_temp,est_p),dT)
print(coeffs,covar)

plt.plot(dt,dT,'o')
plt.plot(dt,dT_model((est_temp,est_p),coeffs[0]))
