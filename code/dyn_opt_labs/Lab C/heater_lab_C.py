import tclab
import numpy as np
import time
import matplotlib.pyplot as plt
import winsound
from scipy.stats import norm
duration = 1000  # millisecond
freq = 440  # Hz


trial=input("trial?")
file_name='Heater_lab_trial_{}.txt'.format(trial)

# Connect to Arduino
a = tclab.TCLab()

# Get Version
print(a.version)

# Turn LED on
print('LED On')
a.LED(100)

# Run time in minutes
run_time = 10.0

# Number of cycles
loops = int(60.0*run_time)
tm = np.zeros(loops)

# Temperature (K)
Tsp1 = np.ones(loops) * 23.0 # set point (degC)
T1 = np.ones(loops) * a.T1 # measured T (degC)

Tsp2 = np.ones(loops) * 23.0 # set point (degC)
T2 = np.ones(loops) * a.T2 # measured T (degC)

# step test (0 - 100%)
Q1 = np.ones(loops) * 0.0
Q2 = np.ones(loops) * 0.0


#rand_times=norm.rvs(2.5,0.5,3)/10*600
step_times_1=[6,120,360,420,500,600]
#step_times_1[0]=6
#for i in range(0,3):
#  step_times_1[i+1]=rand_times[i]+step_times_1[i]
#step_times_1[4]=600
#step_times_1=step_times_1.astype(int)

#rand_times=norm.rvs(2.5,0.5,3)/10*600
step_times_2=[6,130,250,380,490,600]
#step_times_2[0]=6
#for i in range(0,3):
#  step_times_2[i+1]=rand_times[i]+step_times_2[i]
#step_times_2[4]=600
#step_times_2=step_times_2.astype(int)

q1_vals=[80,40,0,60]
q2_vals=[20,50,30,0]

for i in range(4):  
  Q1[step_times_1[i]:step_times_1[i+1]]=q1_vals[i]
  Q2[step_times_2[i]:step_times_2[i+1]]=q2_vals[i]


print('Running Main Loop. Ctrl-C to end.')
print('  Time   Q1     Q2    T1     T2')
print('{:6.1f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}'.format(tm[0], \
                                                       Q1[0], \
                                                       Q2[0], \
                                                       T1[0], \
                                                       T2[0]))

# Create plot
plt.figure(figsize=(10,7))
plt.ion()
plt.show()

# Main Loop
start_time = time.time()
prev_time = start_time
try:
    for i in range(1,loops):
        # Sleep time
        sleep_max = 1.0
        sleep = sleep_max - (time.time() - prev_time)
        if sleep>=0.01:
            time.sleep(sleep)
        else:
            time.sleep(0.01)

        # Record time and change in time
        t = time.time()
        dt = t - prev_time
        prev_time = t
        tm[i] = t - start_time
                    
        # Read temperatures in Kelvin 
        T1[i] = a.T1
        T2[i] = a.T2

        ###############################
        ### CONTROLLER or ESTIMATOR ###
        ###############################

        # Write output (0-100)
        a.Q1(Q1[i])
        a.Q2(Q2[i])

        # Print line of data
        print('{:6.1f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}'.format(tm[i], \
                                                               Q1[i], \
                                                               Q2[i], \
                                                               T1[i], \
                                                               T2[i]))

        # Plot
        plt.clf()
        ax=plt.subplot(2,1,1)
        ax.grid()
        plt.plot(tm[0:i],T1[0:i],'ro',label=r'$T_1$')
        plt.plot(tm[0:i],T2[0:i],'bx',label=r'$T_2$')
        plt.ylabel('Temperature (degC)')
        plt.legend(loc='best')
        ax=plt.subplot(2,1,2)
        ax.grid()
        plt.plot(tm[0:i],Q1[0:i],'r-',label=r'$Q_1$')
        plt.plot(tm[0:i],Q2[0:i],'b:',label=r'$Q_2$')
        plt.ylabel('Heaters')
        plt.xlabel('Time (sec)')
        plt.legend(loc='best')
        plt.draw()
        plt.pause(0.05)

    # Turn off heaters
    a.Q1(0)
    a.Q2(0)
    a.LED(0)
    # Save text file
    a.save_txt(file_name,tm[0:i],Q1[0:i],Q2[0:i],T1[0:i],T2[0:i],Tsp1[0:i],Tsp2[0:i])
    # Save figure
    plt.savefig('Heater_lab_trial_{}.png'.format(trial))
    winsound.Beep(freq, duration)
        
# Allow user to end loop with Ctrl-C           
except KeyboardInterrupt:
    # Disconnect from Arduino
    a.Q1(0)
    a.Q2(0)
    a.LED(0)
    print('Shutting down')
    a.close()
    a.save_txt(file_name,tm[0:i],Q1[0:i],Q2[0:i],T1[0:i],T2[0:i],Tsp1[0:i],Tsp2[0:i])
    plt.savefig('Heater_lab_trial_{}.png'.format(trial))
    
# Make sure serial connection still closes when there's an error
except:           
    # Disconnect from Arduino
    a.Q1(0)
    a.Q2(0)
    a.LED(0)
    print('Error: Shutting down')
    a.close()
    a.save_txt(file_name,tm[0:i],Q1[0:i],Q2[0:i],T1[0:i],T2[0:i],Tsp1[0:i],Tsp2[0:i])
    plt.savefig('Heater_lab_trial_{}.png'.format(trial))
    winsound.Beep(freq, duration)
    raise
