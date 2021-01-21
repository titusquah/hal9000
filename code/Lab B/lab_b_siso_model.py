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



#n=60*10+1 #10 min time points


#-2.06486958e-01 -1.85054469e-04

m_data=pd.read_excel("heater_lab_averages.xlsx")
#
#
m_time=np.asarray(m_data["Time"])
mQ1=np.asarray(m_data["Q1"])
mT1=np.asarray(m_data["T1"])+273.15
mQ2=np.asarray(m_data["Q2"])
mT2=np.asarray(m_data["T2"])+273.15

m.time=m_time#np.linspace(0,n-1,n) #time vector
n=len(m.time)

step=-1
places=1
diff=-1e10
r_21=0
r_22=0
r_tot=0

#u_val=6.
u_val1=5
u_val2=19

#p1=np.zeros(n)
#p2=np.zeros(n)
#
#p1[int(60*0.1):]=1
#p2[300:]=0.75


Q1=m.Param(value=mQ1) #Percent Heater (0-100%)
Q2=m.Param(value=mQ2)
T1_0=m.Param(value=mT1[0])#+273.15)      #Init temp K
T2_0=m.Param(value=mT2[0])#+273.15)
Ta=m.Param(value=np.mean([mT1[0],mT2[0]]))#+273.15)      #Surr Temp


U1=m.Param(value=u_val1)              #W/m^2 K
U2=m.Param(value=u_val2)
mass=m.Param(value=4./1000.)      #kg
Cp=m.Param(value=0.5*1e3)         #J/kgK
A=m.Param(value=10./100**2)       #Area in m^2
alpha1 = m.Param(value=0.01)       #W / % heater
alpha2= m.Param(value=0.015)       #W / % heater
eps = m.Param(value=0.9)          #Emissivity
A_s=m.Param(value=2./100**2)      #Survce area between heat sinks in m^2
sigma = m.Const(5.67e-8)          #Stefan boltzman constant




T1=m.Var(value=T1_0)
T2=m.Var(value=T2_0)

m.Equation(mass*Cp*T1.dt()==U1*A*(Ta-T1)+eps*sigma*A*(Ta**4-T1**4)+Q1*alpha1+U1*A_s*(T2-T1)+eps*sigma*A_s*(T2**4-T1**4))
m.Equation(mass*Cp*T2.dt()==U2*A*(Ta-T2)+eps*sigma*A*(Ta**4-T2**4)+Q2*alpha2+U2*A_s*(T1-T2)+eps*sigma*A_s*(T1**4-T2**4))
m.options.IMODE=4
m.solve(disp=False)


#num=sum((mT-T.value)**2)
#den=sum((mT-np.mean(mT))**2)
#r_2=(1-num/den)
#
cum_err=np.sum(np.abs(mT1-T1.value))+np.sum(np.abs(mT2-T2.value))

plt.close('all')
plt.figure(1)
plt.subplot(211)
plt.plot(m.time,T1.value,'r-',label='T1')
plt.plot(m.time,T2.value,'k-',label='T2')
plt.plot(m_time,mT1,'ro',label="Measured T1")
plt.plot(m_time,mT2,'ok',label="Measured T2")
plt.legend(loc="best",prop={'size': 18})
plt.xlabel("",fontsize=18)
plt.ylabel("Temperature (K)",fontsize=18)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.yticks(fontsize=16)

plt.subplot(212)
plt.plot(m_time,mQ1/100*3,'r',label="Q1")
plt.plot(m_time,mQ2/100*3,'k',label="Q2")
plt.legend(loc="best",prop={'size': 18})
plt.xlabel("Time",fontsize=18)
plt.ylabel("Heater Output (W)",fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


#print(r_21,r_22)
print(cum_err)



#for place in range(places):
#  while diff<0:
#    print(u_val1)
#    #Parameters
#    Q1=m.Param(value=mQ1) #Percent Heater (0-100%)
#    Q2=m.Param(value=mQ2)
#    T1_0=m.Param(value=mT1[0])#+273.15)      #Init temp K
#    T2_0=m.Param(value=mT2[0])#+273.15)
#    Ta=m.Param(value=np.mean([mT1[0],mT2[0]]))#+273.15)      #Surr Temp
#    
#    U1=m.Param(value=u_val1)              #W/m^2 K
#    U2=m.Param(value=u_val2)              #W/m^2 K
#    mass=m.Param(value=4./1000.)      #kg
#    Cp=m.Param(value=0.5*1e3)         #J/kgK
#    A=m.Param(value=10./100**2)       #Area in m^2
#    alpha1 = m.Param(value=0.01)       #W / % heater
#    alpha2= m.Param(value=0.015)       #W / % heater
#    eps = m.Param(value=0.9)          #Emissivity
#    A_s=m.Param(value=2./100**2)      #Survce area between heat sinks in m^2
#    sigma = m.Const(5.67e-8)          #Stefan boltzman constant
#    
#    
#    
#    
#    T1=m.Var(value=T1_0)
#    T2=m.Var(value=T2_0)
#    
#    m.Equation(mass*Cp*T1.dt()==U1*A*(Ta-T1)+eps*sigma*A*(Ta**4-T1**4)+Q1*alpha1+U2*A_s*(T2-T1)+eps*sigma*A_s*(T2**4-T1**4))
#    m.Equation(mass*Cp*T2.dt()==U2*A*(Ta-T2)+eps*sigma*A*(Ta**4-T2**4)+Q2*alpha2+U2*A_s*(T1-T2)+eps*sigma*A_s*(T1**4-T2**4))
#    m.options.IMODE=4
#    m.solve(disp=False)
#    
#    
#    num1=sum((mT1-T1.value)**2)
#    den1=sum((mT1-np.mean(mT1))**2)
#    diff=r_21-(1-num1/den1)
#    r_21=(1-num1/den1)
#    
#    num2=sum((mT2-T2.value)**2)
#    den2=sum((mT2-np.mean(mT2))**2)
#  #  diff2=r_22-(1-num2/den2)
#    r_22=(1-num2/den2)
##    
##    diff=r_tot-np.mean([r_21,r_22])
#    r_tot=np.mean([r_21,r_22])
#    
#    
#    
#    u_val1+=step
#    print(r_21,r_22,r_tot)
#    
##  diff=-1e4
##  u_val1-=2*step
##  step/=-10
##  u_val_up=u_val1+step
##  u_val_down=u_val1-step
##  
##  #Parameters
##  Q1=m.Param(value=mQ1) #Percent Heater (0-100%)
##  Q2=m.Param(value=mQ2)
##  T1_0=m.Param(value=mT1[0])#+273.15)      #Init temp K
##  T2_0=m.Param(value=mT2[0])#+273.15)
##  Ta=m.Param(value=np.mean([mT1[0],mT2[0]]))#+273.15)      #Surr Temp
##  
##  U1=m.Param(value=u_val1)              #W/m^2 K
##  U2=m.Param(value=u_val2)              #W/m^2 K
##  
##  mass=m.Param(value=4./1000.)      #kg
##  Cp=m.Param(value=0.5*1e3)         #J/kgK
##  A=m.Param(value=10./100**2)       #Area in m^2
##  alpha1 = m.Param(value=0.01)       #W / % heater
##  alpha2= m.Param(value=0.015)       #W / % heater
##
##  eps = m.Param(value=0.9)          #Emissivity
##  A_s=m.Param(value=2./100**2)      #Survce area between heat sinks in m^2
##  sigma = m.Const(5.67e-8)          #Stefan boltzman constant
##  
##  
##  
##  
##  T1=m.Var(value=T1_0)
##  T2=m.Var(value=T2_0)
##  
##  m.Equation(mass*Cp*T1.dt()==U1*A*(Ta-T1)+eps*sigma*A*(Ta**4-T1**4)+Q1*alpha1+U1*A_s*(T2-T1)+eps*sigma*A_s*(T2**4-T1**4))
##  m.Equation(mass*Cp*T2.dt()==U2*A*(Ta-T2)+eps*sigma*A*(Ta**4-T2**4)+Q2*alpha2+U2*A_s*(T1-T2)+eps*sigma*A_s*(T1**4-T2**4))
##  m.options.IMODE=4
##  m.solve(disp=False)
##  
##  
##  num1=sum((mT1-T1.value)**2)
##  den1=sum((mT1-np.mean(mT1))**2)
###  diff=r_21-(1-num1/den1)
##  r_21=(1-num1/den1)
##  
##  num2=sum((mT2-T2.value)**2)
##  den2=sum((mT2-np.mean(mT2))**2)
###  diff2=r_22-(1-num2/den2)
##  r_22=(1-num2/den2)
##  
##  r_2_up=r_21
##  
##  #Parameters
##  Q1=m.Param(value=mQ1) #Percent Heater (0-100%)
##  Q2=m.Param(value=mQ2)
##  T1_0=m.Param(value=mT1[0])#+273.15)      #Init temp K
##  T2_0=m.Param(value=mT2[0])#+273.15)
##  Ta=m.Param(value=np.mean([mT1[0],mT2[0]]))#+273.15)      #Surr Temp
##  
##  U1=m.Param(value=u_val1)              #W/m^2 K
##  U2=m.Param(value=u_val2)              #W/m^2 K  
##  mass=m.Param(value=4./1000.)      #kg
##  Cp=m.Param(value=0.5*1e3)         #J/kgK
##  A=m.Param(value=10./100**2)       #Area in m^2
##  alpha1 = m.Param(value=0.01)       #W / % heater
##  alpha2= m.Param(value=0.015)       #W / % heater
##  
##  eps = m.Param(value=0.9)          #Emissivity
##  A_s=m.Param(value=2./100**2)      #Survce area between heat sinks in m^2
##  sigma = m.Const(5.67e-8)          #Stefan boltzman constant
##  
##  
##  
##  
##  T1=m.Var(value=T1_0)
##  T2=m.Var(value=T2_0)
##  
##  m.Equation(mass*Cp*T1.dt()==U1*A*(Ta-T1)+eps*sigma*A*(Ta**4-T1**4)+Q1*alpha1+U1*A_s*(T2-T1)+eps*sigma*A_s*(T2**4-T1**4))
##  m.Equation(mass*Cp*T2.dt()==U2*A*(Ta-T2)+eps*sigma*A*(Ta**4-T2**4)+Q2*alpha2+U2*A_s*(T1-T2)+eps*sigma*A_s*(T1**4-T2**4))
##  m.options.IMODE=4
##  m.solve(disp=False)
##  
##  
##  num1=sum((mT1-T1.value)**2)
##  den1=sum((mT1-np.mean(mT1))**2)
###  diff1=r_21-(1-num1/den1)
##  r_21=(1-num1/den1)
##  
##  num2=sum((mT2-T2.value)**2)
##  den2=sum((mT2-np.mean(mT2))**2)
###  diff2=r_22-(1-num2/den2)
##  r_22=(1-num2/den2)
##
##  r_2_down=r_21
##  
##  if r_2_up>r_2_down:
##    step=step
##  elif r_2_up<r_2_down:
##    step=-step
##  
##  u_val1+=step
#u_val1=3
#diff=-1e4
#r_22=0
#step=1
#for place in range(places):
#  while diff<0:
#    print(u_val2)
#    #Parameters
#    Q1=m.Param(value=mQ1) #Percent Heater (0-100%)
#    Q2=m.Param(value=mQ2)
#    T1_0=m.Param(value=mT1[0])#+273.15)      #Init temp K
#    T2_0=m.Param(value=mT2[0])#+273.15)
#    Ta=m.Param(value=np.mean([mT1[0],mT2[0]]))#+273.15)      #Surr Temp
#    
#    U1=m.Param(value=u_val1)              #W/m^2 K
#    U2=m.Param(value=u_val2)              #W/m^2 K
#    mass=m.Param(value=4./1000.)      #kg
#    Cp=m.Param(value=0.5*1e3)         #J/kgK
#    A=m.Param(value=10./100**2)       #Area in m^2
#    alpha1 = m.Param(value=0.01)       #W / % heater
#    alpha2= m.Param(value=0.015)       #W / % heater  
#
#    eps = m.Param(value=0.9)          #Emissivity
#    A_s=m.Param(value=2./100**2)      #Survce area between heat sinks in m^2
#    sigma = m.Const(5.67e-8)          #Stefan boltzman constant
#    
#    
#    
#    
#    T1=m.Var(value=T1_0)
#    T2=m.Var(value=T2_0)
#    
#    m.Equation(mass*Cp*T1.dt()==U1*A*(Ta-T1)+eps*sigma*A*(Ta**4-T1**4)+Q1*alpha1+U2*A_s*(T2-T1)+eps*sigma*A_s*(T2**4-T1**4))
#    m.Equation(mass*Cp*T2.dt()==U2*A*(Ta-T2)+eps*sigma*A*(Ta**4-T2**4)+Q2*alpha2+U2*A_s*(T1-T2)+eps*sigma*A_s*(T1**4-T2**4))
#    m.options.IMODE=4
#    m.solve(disp=False)
#    
#    
#    num1=sum((mT1-T1.value)**2)
#    den1=sum((mT1-np.mean(mT1))**2)
#  #  diff1=r_21-(1-num1/den1)
#    r_21=(1-num1/den1)
#    
#    num2=sum((mT2-T2.value)**2)
#    den2=sum((mT2-np.mean(mT2))**2)
#    diff=r_22-(1-num2/den2)
#    r_22=(1-num2/den2)
#    
##    diff=r_tot-np.mean([r_21,r_22])
#    r_tot=np.mean([r_21,r_22])
#    
#    
#    
#    u_val2+=step
#    print(r_21,r_22,r_tot)
#    
##  diff=-1e4
##  u_val2-=2*step
##  step/=-10
##  u_val_up=u_val2+step
##  u_val_down=u_val2-step
##  
##  #Parameters
##  Q1=m.Param(value=mQ1) #Percent Heater (0-100%)
##  Q2=m.Param(value=mQ2)
##  T1_0=m.Param(value=mT1[0])#+273.15)      #Init temp K
##  T2_0=m.Param(value=mT2[0])#+273.15)
##  Ta=m.Param(value=np.mean([mT1[0],mT2[0]]))#+273.15)      #Surr Temp
##  
##  U1=m.Param(value=u_val1)              #W/m^2 K
##  U2=m.Param(value=u_val2)              #W/m^2 K
##  
##  mass=m.Param(value=4./1000.)      #kg
##  Cp=m.Param(value=0.5*1e3)         #J/kgK
##  A=m.Param(value=10./100**2)       #Area in m^2
##  alpha1 = m.Param(value=0.01)       #W / % heater
##  alpha2= m.Param(value=0.015)       #W / % heater
##  
##  eps = m.Param(value=0.9)          #Emissivity
##  A_s=m.Param(value=2./100**2)      #Survce area between heat sinks in m^2
##  sigma = m.Const(5.67e-8)          #Stefan boltzman constant
##  
##  
##  
##  
##  T1=m.Var(value=T1_0)
##  T2=m.Var(value=T2_0)
##  
##  m.Equation(mass*Cp*T1.dt()==U1*A*(Ta-T1)+eps*sigma*A*(Ta**4-T1**4)+Q1*alpha1+U1*A_s*(T2-T1)+eps*sigma*A_s*(T2**4-T1**4))
##  m.Equation(mass*Cp*T2.dt()==U2*A*(Ta-T2)+eps*sigma*A*(Ta**4-T2**4)+Q2*alpha2+U2*A_s*(T1-T2)+eps*sigma*A_s*(T1**4-T2**4))
##  m.options.IMODE=4
##  m.solve(disp=False)
##  
##  
##  num1=sum((mT1-T1.value)**2)
##  den1=sum((mT1-np.mean(mT1))**2)
###  diff1=r_21-(1-num1/den1)
##  r_21=(1-num1/den1)
##  
##  num2=sum((mT2-T2.value)**2)
##  den2=sum((mT2-np.mean(mT2))**2)
##  diff=r_22-(1-num2/den2)
##  r_22=(1-num2/den2)
##  
##  r_2_up=r_22
##  
##  #Parameters
##  Q1=m.Param(value=mQ1) #Percent Heater (0-100%)
##  Q2=m.Param(value=mQ2)
##  T1_0=m.Param(value=mT1[0])#+273.15)      #Init temp K
##  T2_0=m.Param(value=mT2[0])#+273.15)
##  Ta=m.Param(value=np.mean([mT1[0],mT2[0]]))#+273.15)      #Surr Temp
##  
##  U1=m.Param(value=u_val1)              #W/m^2 K
##  U2=m.Param(value=u_val2)              #W/m^2 K  
##  mass=m.Param(value=4./1000.)      #kg
##  Cp=m.Param(value=0.5*1e3)         #J/kgK
##  A=m.Param(value=10./100**2)       #Area in m^2
##  alpha1 = m.Param(value=0.01)       #W / % heater
##  alpha2= m.Param(value=0.015)       #W / % heater
##  
##  eps = m.Param(value=0.9)          #Emissivity
##  A_s=m.Param(value=2./100**2)      #Survce area between heat sinks in m^2
##  sigma = m.Const(5.67e-8)          #Stefan boltzman constant
##  
##  
##  
##  
##  T1=m.Var(value=T1_0)
##  T2=m.Var(value=T2_0)
##  
##  m.Equation(mass*Cp*T1.dt()==U1*A*(Ta-T1)+eps*sigma*A*(Ta**4-T1**4)+Q1*alpha1+U1*A_s*(T2-T1)+eps*sigma*A_s*(T2**4-T1**4))
##  m.Equation(mass*Cp*T2.dt()==U2*A*(Ta-T2)+eps*sigma*A*(Ta**4-T2**4)+Q2*alpha2+U2*A_s*(T1-T2)+eps*sigma*A_s*(T1**4-T2**4))
##  m.options.IMODE=4
##  m.solve(disp=False)
##  
##  
##  num1=sum((mT1-T1.value)**2)
##  den1=sum((mT1-np.mean(mT1))**2)
###  diff1=r_21-(1-num1/den1)
##  r_21=(1-num1/den1)
##  
##  num2=sum((mT2-T2.value)**2)
##  den2=sum((mT2-np.mean(mT2))**2)
###  diff2=r_22-(1-num2/den2)
##  r_22=(1-num2/den2)
##
##  r_2_down=r_22
##  
##  if r_2_up>r_2_down:
##    step=step
##  elif r_2_up<r_2_down:
##    step=-step
##  
##  u_val2+=step

winsound.Beep(freq, duration)