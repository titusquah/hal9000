# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:54:12 2019

@author: Titus
"""

import numpy as np

import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit

from scipy import stats
import uncertainties as unc



def band_predictor(func,x,x_data,y_data,coefficients, confidence_interval=0.95):
  alpha=1-confidence_interval
  data_n=x_data.size
  var_n=len(coefficients)
  quantile=stats.t.ppf(1-alpha/2,data_n-var_n)
  
  se=np.sqrt(1/(data_n-var_n)*np.sum((y_data-func(x_data,*coefficients))**2))
  
  sx=(x-x_data.mean())**2
  sxd=np.sum((x_data-x_data.mean())**2)
  yp=func(x,*coefficients)
  
  dy=quantile*se*np.sqrt(1+(1/data_n)+(sx/sxd))
  lpb,upb=yp-dy,yp+dy
  return lpb,upb

data=np.genfromtxt("Heater_lab_trial_1.txt",delimiter=',')
#0=time (1), 1=heat input 1(%), 2= heat input 2 (%), 3=temp 1 (deg C), 4=temp 2 (deg C), 5=sp 1, 6=sp 2

t=data[:,0]
Q1=data[:,1]
Q2=data[:,2]
T1_meas=data[:,3]
T2_meas=data[:,4]

n=len(t)

def dT_dt(x,t,Q1,Q2,p):

    U,alpha1,alpha2 = p
    
    Ta = 23 + 273.15   # K
    m = 4.0/1000.0     # kg
    Cp = 0.5 * 1000.0  # J/kg-K  
    A = 10.0 / 100.0**2 # Area in m^2
    As = 2.0 / 100.0**2 # Area in m^2
    eps = 0.9          # Emissivity
    sigma = 5.67e-8    # Stefan-Boltzman

    # Temperature States 
    T1 = x[0] + 273.15
    T2 = x[1] + 273.15

    dT1dt = (1/(m*Cp))*(U*A*(Ta-T1)+eps*sigma*A*(Ta**4-T1**4)+Q1*alpha1+U*As*(T2-T1)+eps*sigma*As*(T2**4-T1**4))
    dT2dt = (1/(m*Cp))*(U*A*(Ta-T2)+eps*sigma*A*(Ta**4-T2**4)+Q2*alpha2+U*As*(T1-T2)+eps*sigma*As*(T1**4-T2**4)) 

    return [dT1dt,dT2dt]
  
def temp_integrator(tm,U,alpha1,alpha2):
  T=np.zeros((n,2))
  T[0,0]=T1_meas[0]
  T[0,1]=T1_meas[0]
  T_0=T[0]
  p=(U,alpha1,alpha2)
  for i in range(n-1):
    dt=[t[i],t[i+1]]
    y=odeint(dT_dt,T_0,dt,args=(Q1[i],Q2[i],p))
    T_0=y[-1]
    T[i+1]=T_0
  z=np.empty((n*2))
  z[:n]=T[:,0]
  z[n:]=T[:,1]
  return z


def temp_integrator_for_plots(p):
  T = np.zeros((n,2))
  T[0,0] = T1_meas[0]
  T[0,1] = T2_meas[0]    
  T0 = T[0]
  for i in range(n-1):
      ts = [t[i],t[i+1]]
      y = odeint(dT_dt,T0,ts,args=(Q1[i],Q2[i],p))
      T0 = y[-1]
      T[i+1] = T0
  return T
  
# Parameter initial guess
U = 10.0           # Heat transfer coefficient (W/m^2-K)
alpha1 = 0.0100    # Heat gain 1 (W/%)
alpha2 = 0.0075    # Heat gain 2 (W/%)
  

pinit = [U,alpha1,alpha2]

x=[]
y=np.empty((n*2))
y[:n] = T1_meas
y[n:] = T2_meas

popt, pcov = curve_fit(temp_integrator,x, y)

Uu, alpha1u, alpha2u = unc.correlated_values(popt, pcov)



lpb,upb=band_predictor(temp_integrator,y,y,y,popt)

lpb1=lpb[:n]
upb1=upb[:n]

lpb2=lpb[n:]
upb2=upb[n:]

guess_T1=temp_integrator_for_plots(pinit)[:,0]
guess_T2=temp_integrator_for_plots(pinit)[:,1]

opt_T1=temp_integrator_for_plots(popt)[:,0]
opt_T2=temp_integrator_for_plots(popt)[:,1]

print('Optimal Parameters with Uncertanty Range')
print('U: ' + str(Uu))
print('alpha1: ' + str(alpha1u))
print('alpha2: ' + str(alpha2u))
print()
print("Sum of absolute error")
print("T1: {0:.3f}".format(sum(abs(opt_T1-T1_meas))))
print("T2: {0:.3f}".format(sum(abs(opt_T2-T2_meas))))

plt.close('all')
plt.subplot(311)
plt.plot(t,lpb1,'r--',label='')
plt.plot(t,upb1,'r--',label='')
plt.plot(t,opt_T1,'r-',label='Model w/ Optimized Params and Prediction bands')
plt.plot(t,T1_meas,'b-',label='Measured Temp 1')
plt.plot(t,guess_T1,'k-',label="Guess Model")
plt.ylabel('Temperature ($^o$C)')
plt.grid()
plt.legend(loc='best')

plt.subplot(312)
plt.plot(t,lpb2,'r--',label='')
plt.plot(t,upb2,'r--',label='')
plt.plot(t,opt_T2,'r-',label='Model w/ Optimized Params and Prediction bands')
plt.plot(t,T2_meas,'b-',label='Measured Temp 2')
plt.plot(t,guess_T2,'k-',label="Guess Model")
plt.ylabel('Temperature ($^o$C)')
plt.grid()
plt.legend(loc='best')

plt.subplot(313)
plt.plot(t,Q1,label='Heat input 1')
plt.plot(t,Q2,label='Heat input 2')
plt.ylabel('Heat Input (%)')
plt.grid()
plt.legend(loc='best')

  