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

fname='{0}-{1}_{2}-{3}-{4}_'.format(time.localtime().tm_hour,time.localtime().tm_min,time.localtime().tm_mon,time.localtime().tm_mday,time.localtime().tm_year)
# save txt file
def save_txt(t,Q1,Q2,T1,T2):
    data = np.vstack((t,Q1,Q2,T1,T2))  # vertical stack
    data = data.T                 # transpose data
    top = 'Time (sec), Heater 1, Heater 2, ' \
        + 'Temperature 1, Temperature 2' 
    np.savetxt('{0}MPC_data.txt'.format(fname),data,delimiter=',',header=top,comments='')

# Connect to Arduino
a = TCLab()

# Final time
tf = 20 # min
# number of data points (every 3 seconds)
n = tf * 20 + 1

T_surr=23
maxT1=90.
maxT2=70.
minT=23
seed=25
cyc_switch=60
sp2_delay=20

pc_hor=60
steps=int(pc_hor/3)+1

db1=1
db2=0.5
#%%
# Configure heater levels
# Percent Heater (0-100%)
SP1 = np.ones(n)*T_surr+random.random()
SP2 = np.ones(n)*T_surr+random.random()*25
# Heater random steps every 60 steps
# Alternate steps by SP1 and SP2
#  with rapid, random changes every 60 cycles
random.seed(seed)
for i in range(n):
    if i%cyc_switch==0:
      rand=random.random()-0.3
      if rand>0:
        SP1[i:i+cyc_switch] = SP1[i-1]+random.random()*25
      else:
        SP1[i:i+cyc_switch] = SP1[i-1]-random.random()*3
    if (i+sp2_delay)%cyc_switch==0:
      rand=random.random()-0.2
      if rand>0:
        SP2[i:i+cyc_switch] = SP2[i-1]+random.random()*25
      else:
        SP2[i:i+cyc_switch] = SP2[i-1]-random.random()*3

print(min(SP1),max(SP1))
print(min(SP2),max(SP2))
#%%
# heater 2 initially off
#SP2[0:cyc_switch] = T_surr
# heater 1 off at end (last 50 cycles)
#SP1[-cyc_switch:-1] = T_surr+5
#SP2[-cyc_switch:-1] = T_surr+5
        
# Record initial temperatures (degC)
T1m = (a.T1) * np.ones(n)
T2m = (a.T2) * np.ones(n)
# Store MHE values for plots
Q1s=np.zeros(n)
Q2s=np.zeros(n)

U_est=2.68238*np.ones(n)
tau_est=19.619*np.ones(n)
est_a1=0.00988*np.ones(n)
est_a2=0.0060424*np.ones(n)
#%%MPC model

mpc = GEKKO(name='tclab-mpc')


# 120 second time horizon, 40 steps
mpc.time = np.linspace(0,pc_hor,steps)

# Parameters to Estimate
mpc.U = mpc.FV(value=2.68238)
mpc.tau=mpc.FV(value=19.619)
mpc.a1=mpc.FV(value=0.00988)
mpc.a2=mpc.FV(0.0060424)

#mpc.U.STATUS=0
#mpc.tau.STATUS=0
#mpc.a1.STATUS=0
#mpc.a2.STATUS=0



mpc.U.FSTATUS=1
mpc.tau.FSTATUS=1
mpc.a1.FSTATUS=1
mpc.a2.FSTATUS=1

mpc.Ta=mpc.Param(value=23.+273.15)
mpc.c_p=mpc.Param(value=500)
mpc.area=mpc.Param(value=1e-3)
mpc.area_s=mpc.Param(value=2e-4)
mpc.mass=mpc.Param(value=.004)
mpc.eps=mpc.Param(value=0.9)
mpc.sig=mpc.Param(value=5.67e-8)

# Measured inputs
mpc.Q1 = mpc.MV(value=0)
mpc.Q1.FSTATUS = 0 #not measured
mpc.Q1.STATUS=1
mpc.Q1.DCOST=0.1
mpc.Q1.DMAX=30
mpc.Q1.LOWER=0
mpc.Q1.UPPER=100

mpc.Q2 = mpc.MV(value=0)
mpc.Q2.FSTATUS = 0 #not measured
mpc.Q2.STATUS=1
mpc.Q2.DCOST=0.1
mpc.Q2.DMAX=20
mpc.Q2.LOWER=0
mpc.Q2.UPPER=100

# State variables
mpc.TH1 = mpc.SV(value=T1m[0])
mpc.TH2 = mpc.SV(value=T2m[0])

# Measurements for model alignment
mpc.TC1 = mpc.CV(value=T1m[0],name='tc1')
mpc.TC1.STATUS = 1     # minimize error between simulation and measurement
mpc.TC1.FSTATUS = 1    # receive measurement
mpc.TC1.TR_INIT=1

mpc.TC1.TAU=15


mpc.TC2 = mpc.CV(value=T2m[0],name='tc2')
mpc.TC2.STATUS = 1     # minimize error between simulation and measurement
mpc.TC2.FSTATUS = 1    # receive measurement
mpc.TC2.TR_INIT=1

mpc.TC2.TAU=10


mpc.Ta = mpc.Param(value=T_surr+273.15) # degC

# Heat transfer between two heaters
#DT = mpc.Intermediate(TH2-TH1)

# Empirical correlations
mpc.T1=mpc.Intermediate(mpc.TH1+273.15)
mpc.T2=mpc.Intermediate(mpc.TH2+273.15)



mpc.conv=mpc.Intermediate(mpc.U*mpc.area_s*(mpc.T2-mpc.T1))
mpc.rad=mpc.Intermediate(mpc.eps*mpc.sig*mpc.area_s*(mpc.T2**4-mpc.T1**4))

mpc.Equation(mpc.mass*mpc.c_p*mpc.TH1.dt()==mpc.U*mpc.area*(mpc.Ta-mpc.T1)+mpc.eps*mpc.sig*mpc.area*(
    mpc.Ta**4-mpc.T1**4)+mpc.conv+mpc.rad+mpc.a1*mpc.Q1)

mpc.Equation(mpc.mass*mpc.c_p*mpc.TH2.dt()==mpc.U*mpc.area*(mpc.Ta-mpc.T2)+mpc.eps*mpc.sig*mpc.area*(
    mpc.Ta**4-mpc.T2**4)-mpc.conv-mpc.rad+mpc.a2*mpc.Q2)

mpc.Equation(mpc.tau*mpc.TC1.dt()==-mpc.TC1+mpc.TH1)
mpc.Equation(mpc.tau*mpc.TC2.dt()==-mpc.TC2+mpc.TH2)

# Global Options
mpc.options.IMODE   = 6 # MPC
mpc.options.EV_TYPE = 1 # Objective type
mpc.options.NODES   = 3 # Collocation nodes
mpc.options.SOLVER  = 3 # IPOPT
mpc.options.COLDSTART = 1 # COLDSTART on first cycle
#%% mhe model
mhe=GEKKO()
mhe.time=np.linspace(0,60,21)

mhe.U=mhe.FV(value=2.68,lb=1,ub=6)
mhe.U.DMAX=1
mhe.U.STATUS=0
mhe.U.FSTATUS=0

mhe.tau=mhe.FV(value=19.6,lb=5,ub=25)
mhe.tau.DMAX=2
mhe.tau.STATUS=0
mhe.tau.FSTATUS=0

mhe.a1=mhe.FV(value=0.00988,lb=0.003,ub=0.03)
mhe.a1.DMAX=0.01
mhe.a1.STATUS=0
mhe.a1.FSTATUS=0

mhe.a2=mhe.FV(value=0.0060,lb=0.001,ub=0.02)
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
#%%
##################################################################
# Create plot
plt.close('all')
plt.figure(figsize=(10,7))
plt.ion()
plt.show()

# Main Loop
start_time = time.time()
prev_time = start_time
tm = np.zeros(n)
a.LED(100)
try:
    for i in range(1,n):
        # Sleep time
        sleep_max = 3.0
        sleep = sleep_max - (time.time() - prev_time)
        if sleep>=0.01:
            time.sleep(sleep-0.01)
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
        

        
        mhe.TC1.MEAS=T1m[i]
        mhe.TC2.MEAS=T2m[i]
        
        mhe.Q1.MEAS=Q1s[i-1]
        mhe.Q2.MEAS=Q2s[i-1]
        
        if i%10==0 and i>9:
          mhe.U.STATUS=1
          mhe.tau.STATUS=1
          mhe.a1.STATUS=1
          mhe.a2.STATUS=1
#        else:
#          mhe.U.STATUS=0
#          mhe.tau.STATUS=0
#          mhe.a1.STATUS=0
#          mhe.a2.STATUS=0
#        
#        if i==30:
#          mhe.U.DMAX=0.5
#          mhe.tau.DMAX=0.5
#          mhe.a1.DMAX=0.0005
#          mhe.a2.DMAX=0.0005
          
        try:
          mhe.solve(disp=False)
          U_est[i]=mhe.U.NEWVAL
          tau_est[i]=mhe.tau.NEWVAL
          est_a1[i]=mhe.a1.NEWVAL
          est_a2[i]=mhe.a2.NEWVAL
        except:
          U_est[i]= U_est[i-1]
          tau_est[i]=tau_est[i-1]
          est_a1[i]=est_a1[i-1]
          est_a2[i]=est_a2[i-1]

       
        # Insert measurements
        mpc.TC1.MEAS = T1m[i]
        mpc.TC2.MEAS = T2m[i]
        
        mpc.TC1.SPHI=SP1[i]+db1
        mpc.TC1.SPLO=SP1[i]-db1

        mpc.TC2.SPHI=SP2[i]+db2
        mpc.TC2.SPLO=SP2[i]-db2
        
        mpc.U.MEAS=U_est[i]
        mpc.tau.MEAS=tau_est[i]
        mpc.a1.MEAS=est_a1[i]
        mpc.a2.MEAS=est_a2[i]
        
        
        # Predict Parameters and Temperatures with MHE
        # use remote=False for local solve\
        try:
          mpc.solve()#disp=False) 
          Q1s[i+1]  = mpc.Q1.NEWVAL
          Q2s[i+1]  = mpc.Q2.NEWVAL
          with open(mpc.path+'//results.json') as f:
              results = json.load(f)
       
        except:
            Q1s[i+1]  = 0
            Q2s[i+1]  = 0
            
        
        # Write new heater values (0-100)
        a.Q1(Q1s[i])
        a.Q2(Q2s[i])

        # Plot
        plt.clf()
        ax=plt.subplot(4,1,1)
        ax.grid()
        plt.plot(tm[0:i+1],SP1[0:i+1]+db1,'k-',label=r'$T_1$ SP')
        plt.plot(tm[0:i+1],SP1[0:i+1]-db1,'k-',label='')
        plt.plot(tm[0:i+1],T1m[0:i+1],'ro',label=r'$T_1$ meas')
        plt.plot(tm[i]+mpc.time,results['tc1.bcv'],'r-',label=r'$T_1$ Pred')
        plt.plot(tm[i]+mpc.time,results['tc1.tr_hi'],'k--',label=r'$T_1$ Traj')
        plt.plot(tm[i]+mpc.time,results['tc1.tr_lo'],'k--',label=r'')
        plt.ylabel('Temperature (degC)')
        plt.legend(loc=2)
        
        ax=plt.subplot(4,1,2)
        ax.grid()
        plt.plot(tm[0:i+1],SP2[0:i+1]+db2,'k-',label=r'$T_2$ SP')
        plt.plot(tm[0:i+1],SP2[0:i+1]-db2,'k-',label='')
        plt.plot(tm[0:i+1],T2m[0:i+1],'bo',label=r'$T_2$ meas')
        plt.plot(tm[i]+mpc.time,results['tc2.bcv'],'b-',label=r'$T_2$ Pred')
        plt.plot(tm[i]+mpc.time,results['tc2.tr_hi'],'k--',label=r'$T_2$ Traj')
        plt.plot(tm[i]+mpc.time,results['tc2.tr_lo'],'k--',label=r'')
        plt.ylabel('Temperature (degC)')
        plt.legend(loc=2)
        
        ax=plt.subplot(4,1,3)
        ax.grid()
        plt.plot([tm[i],tm[i]],[0,100],'k-',label=r'Current Time')
        plt.plot(tm[0:i+1],Q1s[0:i+1],'r-',label=r'$Q_1$')
        plt.plot(tm[0:i+1],Q2s[0:i+1],'b-',label=r'$Q_2$')
        plt.plot(tm[i]+mpc.time,mpc.Q1.value,'r--',label=r'$Q_1$ forecast' )
        plt.plot(tm[i]+mpc.time,mpc.Q2.value,'b--',label=r'$Q_2$ forecast' )
        plt.plot(tm[i]+mpc.time[1],mpc.Q1.value[1],'ro',label=r'' )
        plt.plot(tm[i]+mpc.time[1],mpc.Q2.value[1],'bo-',label=r'' )
        
        plt.ylabel('Heaters')
        plt.xlabel('Time(s)')
        plt.legend(loc='best')

        ax=plt.subplot(4,1,4)
        ax.grid()
        plt.plot(tm[0:i],U_est[0:i],'k-',label='Heat Transfer Coeff')
        plt.plot(tm[0:i],tau_est[0:i],'g:',label=r'$\tau$')        
        plt.plot(tm[0:i],est_a1[0:i]*1000,'r--',label=r'$\alpha_1$x1000')
        plt.plot(tm[0:i],est_a2[0:i]*1000,'b--',label=r'$\alpha_2$x1000')
        plt.ylabel('Parameters')
        plt.legend(loc='best')
        
        plt.draw()
        plt.pause(0.05)
        
    # Turn off heaters
    a.Q1(0)
    a.Q2(0)
    a.LED(0)
    save_txt(tm,Q1s,Q2s,T1m,T2m)
    # Save figure
    plt.savefig('{0}MPC_pic.png'.format(fname))
    a.close()
    
# Allow user to end loop with Ctrl-C           
except KeyboardInterrupt:
    # Disconnect from Arduino
    a.Q1(0)
    a.Q2(0)
    print('Shutting down')
    a.LED(0)
    a.close()
    plt.savefig('{0}pic.png'.format(fname))
    raise
    
# Make sure serial connection still closes when there's an error
except:           
    # Disconnect from Arduino
    a.Q1(0)
    a.Q2(0)
    a.LED(0)
    print('Error: Shutting down')
    a.close()
    plt.savefig('{0}pic.png'.format(fname))
    raise