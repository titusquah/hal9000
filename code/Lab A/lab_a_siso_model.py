# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 14:41:54 2019

@author: Titus
"""

from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import winsound
duration = 1000  # millisecond
freq = 440  # Hz

m=GEKKO()

step=0.001

n=60*10 #10 min time points


#-2.06486958e-01 -1.85054469e-04

m_data=pd.read_excel("average.xlsx")


mp=np.asarray(m_data["Q"])
mT=np.asarray(m_data["T"])+273.15
mt=np.asarray(m_data["t"])

m.time=mt#np.linspace(0,n-1,n) #time vector

#diff=-1e10
#r_2=0
#u_val=10.46
#while diff<0:
#    
#  #Parameters
#  Q=m.Param(value=mp) #Percent Heater (0-100%)
#  T0=m.Param(value=22.189+273.15)      #Init temp 
#  Ta=m.Param(value=22.189+273.15)      #Surr Temp
#  
#  U=m.Param(value=u_val)              #W/m^2 K
#  mass=m.Param(value=4./1000.)      #kg
#  Cp=m.Param(value=0.5*1e3)         #J/kgK
#  A=m.Param(value=12./100**2)       #Area in m^2
#  alpha = m.Param(value=0.01)       #W / % heater
#  eps = m.Param(value=0.9)          #Emissivity
#  sigma = m.Const(5.67e-8)          #Stefan boltzman constant
#  
#  
#  
#  T=m.Var(value=T0)
#  
#  m.Equation(mass*Cp*T.dt()==U*A*(Ta-T)+eps*sigma*A*(Ta-T)**4+Q*alpha)
#  m.options.IMODE=4
#  m.solve(disp=False)
#  
#  
#  num=sum((mT-T.value)**2)
#  den=sum((mT-np.mean(mT))**2)
#  diff=r_2-(1-num/den)
#  r_2=(1-num/den)
#  
#  u_val+=step
#  print(r_2)
#  
u_val=10.468
Q=m.Param(value=mp) #Percent Heater (0-100%)
T0=m.Param(value=22.189+273.15)      #Init temp 
Ta=m.Param(value=22.189+273.15)      #Surr Temp

U=m.Param(value=u_val)              #W/m^2 K
mass=m.Param(value=4./1000.)      #kg
Cp=m.Param(value=0.5*1e3)         #J/kgK
A=m.Param(value=12./100**2)       #Area in m^2
alpha = m.Param(value=0.01)       #W / % heater
eps = m.Param(value=0.9)          #Emissivity
sigma = m.Const(5.67e-8)          #Stefan boltzman constant



T=m.Var(value=T0)

m.Equation(mass*Cp*T.dt()==U*A*(Ta-T)+eps*sigma*A*(Ta**4-T**4)+Q*alpha)
m.options.IMODE=4
m.solve(disp=False)


num=sum((mT-T.value)**2)
den=sum((mT-np.mean(mT))**2)
r_2=(1-num/den)

cum_err=np.sum(np.abs(mT-T.value))

plt.close('all')
plt.plot(m.time,T.value,label='Modeled t')
plt.plot(mt,mT,'o',label="Measured Temperature")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Temperature (K)")
print(u_val)
print(r_2)
print(cum_err)

winsound.Beep(freq, duration)