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

K1_est=0.97*np.ones(n)
K2_est=0.522*np.ones(n)
K3_est=0.492*np.ones(n)
tau12_est=169*np.ones(n)
tau3_est=19*np.ones(n)
#%%mpc
#########################################################
# Initialize Model as Estimator
#########################################################
mpc = GEKKO(name='tclab-mpc')
#m.server = 'http://127.0.0.1' # if local server is installed

# 120 second time horizon, 40 steps
mpc.time = np.linspace(0,pc_hor,steps)

# Parameters to Estimate
mpc.K1 = mpc.FV(value=0.970)
mpc.K2=mpc.FV(value=0.522)
mpc.K3=mpc.FV(value=0.492)
mpc.tau12=mpc.FV(169)
mpc.tau3=mpc.FV(19)

mpc.K1.FSTATUS=1
mpc.K2.FSTATUS=1
mpc.K3.FSTATUS=1
mpc.tau12.FSTATUS=1
mpc.tau3.FSTATUS=1

mpc.K1.STATUS=0
mpc.K2.STATUS=0
mpc.K3.STATUS=0
mpc.tau12.STATUS=0
mpc.tau3.STATUS=0

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


mpc.Ta = mpc.Param(value=T_surr) # degC

# Heat transfer between two heaters
mpc.DT = mpc.Intermediate(mpc.TH2-mpc.TH1)

# Empirical correlations
mpc.Equation(mpc.tau12 * mpc.TH1.dt() + (mpc.TH1-mpc.Ta) == mpc.K1*mpc.Q1 + mpc.K3*mpc.DT)
mpc.Equation(mpc.tau12 * mpc.TH2.dt() + (mpc.TH2-mpc.Ta) == mpc.K2*mpc.Q2 - mpc.K3*mpc.DT)
mpc.Equation(mpc.tau3 * mpc.TC1.dt()  + mpc.TC1 == mpc.TH1)
mpc.Equation(mpc.tau3 * mpc.TC2.dt()  + mpc.TC2 == mpc.TH2)

# Global Options
mpc.options.IMODE   = 6 # MPC
mpc.options.EV_TYPE = 1 # Objective type
mpc.options.NODES   = 3 # Collocation nodes
mpc.options.SOLVER  = 3 # IPOPT
mpc.options.COLDSTART = 1 # COLDSTART on first cycle
#%% mhe model
mhe = GEKKO(name='tclab-mhe')
#m.server = 'http://127.0.0.1' # if local server is installed

# 120 second time horizon, 40 steps
mhe.time = np.linspace(0,120,41)

# Parameters to Estimate
mhe.K1 = mhe.FV(value=0.970)
mhe.K1.STATUS = 0
mhe.K1.FSTATUS = 0
mhe.K1.DMAX = 0.1
mhe.K1.LOWER = 0.1
mhe.K1.UPPER = 1.0

mhe.K2 = mhe.FV(value=0.522)
mhe.K2.STATUS = 0
mhe.K2.FSTATUS = 0
mhe.K2.DMAX = 0.1
mhe.K2.LOWER = 0.1
mhe.K2.UPPER = 1.0

mhe.K3 = mhe.FV(value=0.492)
mhe.K3.STATUS = 0
mhe.K3.FSTATUS = 0
mhe.K3.DMAX = 0.01
mhe.K3.LOWER = 0.1
mhe.K3.UPPER = 1.0

mhe.tau12 = mhe.FV(value=169)
mhe.tau12.STATUS = 0
mhe.tau12.FSTATUS = 0
mhe.tau12.DMAX = 5.0
mhe.tau12.LOWER = 50.0
mhe.tau12.UPPER = 200

mhe.tau3 = mhe.FV(value=19)
mhe.tau3.STATUS = 0
mhe.tau3.FSTATUS = 0
mhe.tau3.DMAX = 1
mhe.tau3.LOWER = 10
mhe.tau3.UPPER = 20

# Measured inputs
mhe.Q1 = mhe.MV(value=0)
mhe.Q1.FSTATUS = 1 # measured

mhe.Q2 = mhe.MV(value=0)
mhe.Q2.FSTATUS = 1 # measured

# State variables
mhe.TH1 = mhe.SV(value=T1m[0])
mhe.TH2 = mhe.SV(value=T2m[0])

# Measurements for model alignment
mhe.TC1 = mhe.CV(value=T1m[0])
mhe.TC1.STATUS = 1     # minimize error between simulation and measurement
mhe.TC1.FSTATUS = 1    # receive measurement
mhe.TC1.MEAS_GAP = 0.1 # measurement deadband gap

mhe.TC2 = mhe.CV(value=T2m[0])
mhe.TC2.STATUS = 1     # minimize error between simulation and measurement
mhe.TC2.FSTATUS = 1    # receive measurement
mhe.TC2.MEAS_GAP = 0.1 # measurement deadband gap

mhe.Ta = mhe.Param(value=22.5) # degC

# Heat transfer between two heaters
mhe.DT = mhe.Intermediate(mhe.TH2-mhe.TH1)

# Empirical correlations
mhe.Equation(mhe.tau12 * mhe.TH1.dt() + (mhe.TH1-mhe.Ta) == mhe.K1*mhe.Q1 + mhe.K3*mhe.DT)
mhe.Equation(mhe.tau12 * mhe.TH2.dt() + (mhe.TH2-mhe.Ta) == mhe.K2*mhe.Q2 - mhe.K3*mhe.DT)
mhe.Equation(mhe.tau3 * mhe.TC1.dt()  + mhe.TC1 == mhe.TH1)
mhe.Equation(mhe.tau3 * mhe.TC2.dt()  + mhe.TC2 == mhe.TH2)

# Global Options
mhe.options.IMODE   = 5 # MHE
mhe.options.EV_TYPE = 1 # Objective type
mhe.options.NODES   = 3 # Collocation nodes
mhe.options.SOLVER  = 3 # IPOPT
mhe.options.COLDSTART = 1 # COLDSTART on first cycle
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
        
        if i%40==0 and i>9:
          mhe.K1.STATUS=1
          mhe.K2.STATUS=1
          mhe.K3.STATUS=1
          mhe.tau12.STATUS=1
          mhe.tau3.STATUS=1
        else:
          mhe.K1.STATUS=0
          mhe.K2.STATUS=0
          mhe.K3.STATUS=0
          mhe.tau12.STATUS=0
          mhe.tau3.STATUS=0
        
        if i==30:
          mhe.K1.DMAX=0.3
          mhe.K2.DMAX=0.2
          mhe.K3.DMAX=0.1
          mhe.tau12.DMAX=10
          mhe.tau3.DMAX=1
          
        try:
          mhe.solve(disp=False)
          K1_est[i]=mhe.K1.NEWVAL
          K2_est[i]=mhe.K2.NEWVAL
          K3_est[i]=mhe.K3.NEWVAL
          tau12_est[i]=mhe.tau12.NEWVAL
          tau3_est[i]=mhe.tau3.NEWVAL
        except:
          K1_est[i]= K1_est[i-1]
          K2_est[i]=K2_est[i-1]
          K3_est[i]=K3_est[i-1]
          tau12_est[i]=tau12_est[i-1]
          tau3_est[i]=tau3_est[i-1]

       
        # Insert measurements
        mpc.TC1.MEAS = T1m[i]
        mpc.TC2.MEAS = T2m[i]
        
        mpc.TC1.SPHI=SP1[i]+db1
        mpc.TC1.SPLO=SP1[i]-db1

        mpc.TC2.SPHI=SP2[i]+db2
        mpc.TC2.SPLO=SP2[i]-db2
        
        mpc.K1.MEAS=K1_est[i]
        mpc.K2.MEAS=K2_est[i]
        mpc.K3.MEAS=K3_est[i]
        mpc.tau12.MEAS=tau12_est[i]
        mpc.tau3.MEAS=tau3_est[i]
        
        
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
        plt.plot(tm[0:i],K1_est[0:i],'k-',label=r'$K_1$')
        plt.plot(tm[0:i],K2_est[0:i],'g:',label=r'$K_2$')        
        plt.plot(tm[0:i],K3_est[0:i],'r--',label=r'$K_3$')
        plt.plot(tm[0:i],tau12_est[0:i]/1000,'b--',label=r'$\tau_{12}$')
        plt.plot(tm[0:i],tau3_est[0:i]/100,'b--',label=r'$\tau_{3}$')
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