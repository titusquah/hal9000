# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 16:06:49 2019

@author: Titus
"""

from gekko import GEKKO
import numpy as np
from tclab import TCLab
import time
import matplotlib.pyplot as plt
import random

pic_name=input("Name of trial? ")

a=TCLab()

tf=20

nt=20*tf+1

set_Q1=np.zeros(nt)
set_Q2=np.zeros(nt)

set_Q1[3:]=70
set_Q1[100:]=0
set_Q1[200:]=50

set_Q2[25:]=50
set_Q2[125:]=70
set_Q2[210:]=25

random.seed(300)
for i in range(225,401):
  if i%60==0:
    set_Q1[i:i+60]=random.randint(0,70)
  if (i+5)%60==0:
    set_Q2[i:i+60]=random.randint(0,70)
T1m=np.ones(nt)*a.T1
T2m=np.ones(nt)*a.T2
T1_est=T1m.copy()
T2_est=T2m.copy()
U_est=10*np.ones(nt)
tau_est=5*np.ones(nt)
est_a1=0.01*np.ones(nt)
est_a2=0.0075*np.ones(nt)
#%% mhe model
mhe=GEKKO()
mhe.time=np.linspace(0,60,21)

mhe.U=mhe.FV(value=4,lb=1,ub=6)
mhe.U.DMAX=1
mhe.U.STATUS=0
mhe.U.FSTATUS=0

mhe.tau=mhe.FV(value=15,lb=5,ub=25)
mhe.tau.DMAX=2
mhe.tau.STATUS=0
mhe.tau.FSTATUS=0

mhe.a1=mhe.FV(value=0.006,lb=0.003,ub=0.03)
mhe.a1.DMAX=0.01
mhe.a1.STATUS=0
mhe.a1.FSTATUS=0

mhe.a2=mhe.FV(value=0.003,lb=0.001,ub=0.02)
mhe.a2.DMAX=0.01
mhe.a2.STATUS=0
mhe.a2.FSTATUS=0

mhe.Q1=mhe.MV(value=0)
mhe.Q1.STATUS=0
mhe.Q1.FSTATUS=1

mhe.Q2=mhe.MV(value=0)
mhe.Q2.STATUS=0
mhe.Q2.FSTATUS=1

mhe.TH1=mhe.SV(value=T1m[0])
mhe.TH2=mhe.SV(value=T2m[0])


mhe.TC1=mhe.CV(value=T1m[0],lb=0,ub=200)
mhe.TC1.FSTATUS=1
mhe.TC1.STATUS=1
mhe.TC1.MEAS_GAP=0.5

mhe.TC2=mhe.CV(value=T2m[0],lb=0,ub=200)
mhe.TC2.FSTATUS=1
mhe.TC2.STATUS=1
mhe.TC2.MEAS_GAP=0.5

mhe.Ta=mhe.Param(value=23.+273.15)
mhe.c_p=mhe.Param(value=500)
mhe.area=mhe.Param(value=1e-3)
mhe.area_s=mhe.Param(value=2e-4)
mhe.mass=mhe.Param(value=.004)
mhe.eps=mhe.Param(value=0.9)
mhe.sig=mhe.Param(value=5.67e-8)


mhe.T1=mhe.Intermediate(mhe.TH1+273.15)
mhe.T2=mhe.Intermediate(mhe.TH2+273.15)

mhe.conv=mhe.Intermediate(mhe.U*mhe.area_s*(mhe.T2-mhe.T1))
mhe.rad=mhe.Intermediate(mhe.eps*mhe.sig*mhe.area_s*(mhe.T2**4-mhe.T1**4))

mhe.Equation(mhe.mass*mhe.c_p*mhe.TH1.dt()==mhe.U*mhe.area*(mhe.Ta-mhe.T1)+mhe.eps*mhe.sig*mhe.area*(
    mhe.Ta**4-mhe.T1**4)+mhe.conv+mhe.rad+mhe.a1*mhe.Q1)

mhe.Equation(mhe.mass*mhe.c_p*mhe.TH2.dt()==mhe.U*mhe.area*(mhe.Ta-mhe.T2)+mhe.eps*mhe.sig*mhe.area*(
    mhe.Ta**4-mhe.T2**4)-mhe.conv-mhe.rad+mhe.a2*mhe.Q2)

mhe.Equation(mhe.tau*mhe.TC1.dt()==-mhe.TC1+mhe.TH1)
mhe.Equation(mhe.tau*mhe.TC2.dt()==-mhe.TC2+mhe.TH2)

mhe.options.IMODE=5
mhe.options.EV_TYPE=1
mhe.options.SOLVER=3
mhe.options.COLDSTART=1

plt.figure()
plt.ion()
plt.show()

start_time=time.time()
prev_time=start_time
tm=np.zeros(nt) 
a.LED(100)

try:
  for i in range(1,nt):
    sleep_max=3.
    sleep=sleep_max-(time.time()-prev_time)
    if sleep>=0.01:
      time.sleep(sleep-0.01)
    else:
      time.sleep(0.01)
      
    t=time.time()
    dt=t-prev_time
    prev_time=t
    tm[i]=t-start_time
    
    T1m[i]=a.T1
    T2m[i]=a.T2
    
    mhe.TC1.MEAS=T1m[i]
    mhe.TC2.MEAS=T2m[i]
    
    mhe.Q1.MEAS=set_Q1[i-1]
    mhe.Q2.MEAS=set_Q2[i-1]
    
    if i%10==0 and i>9:
      mhe.U.STATUS=1
      mhe.tau.STATUS=1
      mhe.a1.STATUS=1
      mhe.a2.STATUS=1
    else:
      mhe.U.STATUS=0
      mhe.tau.STATUS=0
      mhe.a1.STATUS=0
      mhe.a2.STATUS=0
    
    if i==30:
      mhe.U.DMAX=0.5
      mhe.tau.DMAX=0.5
      mhe.a1.DMAX=0.0005
      mhe.a2.DMAX=0.0005
      
      
    mhe.solve(disp=False)
    
    if mhe.options.APPSTATUS==1:
      T1_est[i]=mhe.TC1.MODEL
      T2_est[i]=mhe.TC2.MOdEL
      
      U_est[i]=mhe.U.NEWVAL
      tau_est[i]=mhe.tau.NEWVAL
      est_a1[i]=mhe.a1.NEWVAL
      est_a2[i]=mhe.a2.NEWVAL
    else:
      T1_est[i]=T1_est[i-1]
      T2_est[i]=T2_est[i-1]
      
      U_est[i]= U_est[i-1]
      tau_est[i]=tau_est[i-1]
      est_a1[i]=est_a1[i-1]
      est_a2[i]=est_a2[i-1]
    a.Q1(set_Q1[i])
    a.Q2(set_Q2[i])
    
    plt.clf()
    ax=plt.subplot(3,1,1)
    ax.grid()
    plt.plot(tm[0:i],T1m[0:i],'ro',label=r'$T_1$ measured')
    plt.plot(tm[0:i],T1_est[0:i],'k-',label=r'$T_1$ model')
    plt.plot(tm[0:i],T2m[0:i],'bx',label=r'$T_2$ measured')
    plt.plot(tm[0:i],T2_est[0:i],'k--',label=r'$T_2$ model')
    plt.ylabel('Temperature (degC)')
    plt.legend(loc=2)
    ax=plt.subplot(3,1,2)
    ax.grid()
    plt.plot(tm[0:i],U_est[0:i],'k-',label='Heat Transfer Coeff')
    plt.plot(tm[0:i],tau_est[0:i],'g:',label=r'$\tau$')        
    plt.plot(tm[0:i],est_a1[0:i]*1000,'r--',label=r'$\alpha_1$x1000')
    plt.plot(tm[0:i],est_a2[0:i]*1000,'b--',label=r'$\alpha_2$x1000')
    plt.ylabel('Parameters')
    plt.legend(loc='best')
    ax=plt.subplot(3,1,3)
    ax.grid()
    plt.plot(tm[0:i],set_Q1[0:i],'r-',label=r'$Q_1$')
    plt.plot(tm[0:i],set_Q2[0:i],'b:',label=r'$Q_2$')
    plt.ylabel('Heaters')
    plt.xlabel('Time (sec)')
    plt.legend(loc='best')
    plt.draw()
    plt.pause(0.05)
  a.Q1(0)
  a.Q2(0)
  a.LED(0)
  a.close()
  
  plt.savefig("{}.png".format(pic_name))
except KeyboardInterrupt:
  a.Q1(0)
  a.Q2(0)
  a.LED(0)
  print("KeyboardInterrupt shutdown")
  a.close()
  plt.savefig("{}.png".format(pic_name))
except Exception as e:
  a.Q1(0)
  a.Q2(0)
  a.LED(0)
  print("{} shutdown".format(e))
  a.close()
  plt.savefig("{}.png".format(pic_name))
  raise
