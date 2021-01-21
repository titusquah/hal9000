# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 19:34:21 2019

@author: Titus
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 17:13:07 2019

@author: Titus
"""
import numpy as np

from gekko import GEKKO
import matplotlib.pyplot as plt





data=np.genfromtxt('Heater_lab_trial_1.txt',delimiter=',')

time=data[:,0]
Q1_m=data[:,1]
Q2_m=data[:,2]
T1_m=data[:,3]
T2_m=data[:,4]




m=GEKKO()
m.time=time

kval=0.5
kl=0.1
ku=2
kdmax=0.1

tauval=50
taul=50
tauu=300
taudmax=5

meas_gap=0.1



k1=m.FV(value=kval,lb=kl,ub=ku)
k2=m.FV(value=kval,lb=kl,ub=ku)
k3=m.FV(value=0.1,lb=0.0001,ub=1)

tau12=m.FV(value=tauval,lb=taul,ub=tauu)
tau3=m.FV(value=15,lb=10,ub=80)

k1.STATUS=1
k2.STATUS=1
k3.STATUS=1
tau12.STATUS=1
tau3.STATUS=1

Q1=m.MV(value=Q1_m)
Q2=m.MV(value=Q2_m)

Q1.FSTATUS=1
Q2.FSTATUS=1

Ta=m.Param(value=T1_m[0])

TH1=m.SV(value=T1_m[0])
TH2=m.SV(value=T2_m[0])

TC1=m.CV(value=T1_m)
TC2=m.CV(value=T2_m)

TC1.STATUS=1
TC1.STATUS=1
TC1.MEAS_GAP=meas_gap

TC2.STATUS=1
TC2.FSTATUS=1
TC2.MEAS_GAP=meas_gap

dT=m.Intermediate(TH2-TH1)

m.Equation(tau12*TH1.dt()+(TH1-Ta)==k1*Q1+k3*dT)
m.Equation(tau12*TH2.dt()+(TH2-Ta)==k2*Q2-k3*dT)

m.Equation(tau3*TC1.dt()+TC1==TH1)
m.Equation(tau3*TC2.dt()+TC2==TH2)

m.options.IMODE=5
m.options.EV_TYPE=2
m.options.SOLVER=3

m.solve(disp=False)

#%%
params=np.array([k1,k2,k3,tau12,tau3])

plt.close('all')
plt.figure()
plt.subplot(311)
plt.plot(time,T1_m,'ro',label="Measured Temp 1")
plt.plot(time,TC1,'b',label="Modeled Temp 1")
plt.legend(loc='best',fontsize=16)
plt.grid()
plt.ylabel('Temperature ($^o$ C)',fontsize=16)
plt.yticks(fontsize=14)

plt.subplot(312)
plt.plot(time,T2_m,'ro',label='Measured Temp 2')
plt.plot(time,TC2,'b',label='Modeled Temp 2')
plt.legend(loc='best',fontsize=16)
plt.grid()
plt.ylabel('Temperature ($^o$ C)',fontsize=16)
plt.yticks(fontsize=14)

plt.subplot(313)
plt.plot(time,Q1_m,'r-',label='Heater 1 output')
plt.plot(time,Q2_m,'b--',label='Heater 2 output')
plt.legend(loc='best',fontsize=16)
plt.grid()
plt.ylabel('Heater output (%)',fontsize=16)
plt.xlabel('Time (s)',fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.show()

#subplots=np.arange(511,516)
#param_names=['$K_1$','$K_2$','$K_3$','$\\tau_{12}$','$\\tau_3$']
#
#plt.figure()
#for i in range(5):
#  plt.subplot(subplots[i])
#  plt.plot(time,params[i],label=param_names[i])
#  plt.grid()
#  plt.legend(loc='best')
#plt.show()

