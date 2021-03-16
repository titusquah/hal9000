from fan_tclab_gym import FanTempControlLabBlackBox as bb_process
from fan_tclab_gym import FanTempControlLabGrayBox as gb_process
from fan_tclab_gym import FanTempControlLabLinearBlackBox as bb_process2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.close()
file_path = r"C:\Users\kervi\Downloads\test4_big_fan.csv"
df = pd.read_csv(file_path)

start = 0
stop = len(df.temp)
d_traj = df.fan_pwm[start:stop] * 100

# d_traj = np.ones(len(d_traj)) * 100

h_traj = df.heater_pwm[start:stop]

def parameters(y,c1a,c2a,c3a,c4a):
	model = bb_process(initial_temp=297.6, #22,
	                   amb_temp=297.6, #22,
	                   dt=0.155,
	                   max_time=stop-1,
	                   d_traj=d_traj,
	                   temp_lb=min(df.temp),#296.15,
					   c1=c1a,
					   c2=c2a,
					   c3=c3a,
					   c4=c4a)
#	                   c1=0.001,
#	                   c2=0.6,
#	                   c3=1e-2,
#	                   c4=0.05)
	
	actions = [0]
	dists = [0]
	states = []
	state = model.reset()
	states.append(state)
	done = False
	ind1 = 0
#	state, reward, done, info = model.step([h_traj[ind1]/100])
#	state, reward, done, info = model.step([0.5])
	#return state[:,0]
	
	while not done:
	    state, reward, done, info = model.step([h_traj[ind1]/100])
	    actions.append(h_traj[ind1])
	    # state, reward, done, info = model.step([0.5])
	    # actions.append(0.5)
	    dists.append(info['dist'])
	    states.append(state)
	    ind1 += 1
	states = np.array(states)
	return states[:,0]-273.15

def parameters2(y,c1a,c2a,c3a,c4a):
	model = bb_process2(initial_temp=297.6, #22,
                   amb_temp=297.6, #22,
                   dt=0.155,
                   max_time=stop,
                   d_traj=d_traj,
                   temp_lb=min(df.temp),#296.15,
				   c1=c1a,
				   c2=c2a,
				   c3=c3a,
				   c4=c4a)
#	                   c1=0.001,
#	                   c2=0.6,
#	                   c3=1e-2,
#	                   c4=0.05)

	actions = [0]
	dists = [0]
	states = []
	state = model.reset()
	states.append(state)
	done = False
	ind1 = 0
#	state, reward, done, info = model.step([h_traj[ind1]/100])
#	state, reward, done, info = model.step([0.5])
	#return state[:,0]
	
	while not done:
	    state, reward, done, info = model.step([h_traj[ind1]/100])
	    actions.append(h_traj[ind1])
	    # state, reward, done, info = model.step([0.5])
	    # actions.append(0.5)
	    dists.append(info['dist'])
	    states.append(state)
	    ind1 += 1
	states = np.array(states)
	return states[:,0]-273.15

#params2=curve_fit(parameters2,df.time[start:stop],df.temp[start:stop],p0=(0.001,.1,1e-3,0.001))

params1=curve_fit(parameters,df.time[start:stop],df.temp[start:stop],p0=(0.001,0.6,1e-2,0.05))

R2=0
guess1=0.001
guess2=0.6
guess3=1e-2
guess4=0.05
i=0

SStot=np.zeros(stop)
SSres=np.zeros(stop)
print('hi')
'''
while R2<0.99 or i<10000:
	params1=curve_fit(parameters,df.time[start:stop],df.temp[start:stop],
						  p0=(guess1,guess2,guess3,guess4))
	model = bb_process(initial_temp=22#297.6,#296.15,
					   amb_temp=22#297.6,#296.15,
	                   dt=1,#0.155,
	                   max_time=6000,
	                   d_traj=d_traj,
	                   temp_lb=min(df.temp),#296.15,
					   c1=params1[0][0],
					   c2=params1[0][1],
					   c3=params1[0][2],
					   c4=params1[0][3])
#	                   c1=0.001,
#	                   c2=0.6,
#	                   c3=1e-2,
#	                   c4=0.05)

	actions = [0]
	dists = [0]
	states = []
	state = model.reset()
	states.append(state)
	done = False
	ind1 = 0
	state, reward, done, info = model.step([h_traj[ind1]/100])
	state, reward, done, info = model.step([0.5])
	while not done:	
		    state, reward, done, info = model.step([h_traj[ind1]/100])
		    actions.append(h_traj[ind1])
		    # state, reward, done, info = model.step([0.5])
		    # actions.append(0.5)
		    dists.append(info['dist'])
		    states.append(state)
		    ind1 += 1
	
	#states = np.array(states)-273.15	
	
	#print(states[i,0])
	ymean=sum(states[:,0])/len(states[:,0])
	for j in range (0,len(states)):
		#print(len(states))
		SStot1=int((df.temp[j]-ymean))**2
		SSres1=(states[j][0]-df.temp[j])**2
		SStot[j]=SStot1
		SSres[j]=SSres1
		
	SStot=sum(SStot)
	SSres=sum(SSres)
	i+=1
	R2=1-(SSres/SStot)
	print(R2)
	guess1=guess1-0.001
	guess2=guess2+0.05
	guess3=guess3+0.0005
	guess4=guess4-0.001
	R2adj=1-(((1-R2)*(len(states)-1))/(len(states)-4-1))
	print(R2adj)
	
'''

model = bb_process(initial_temp=297.6,#296.15,
	                   amb_temp=297.6,#296.15,
	                   dt=0.155,
	                   max_time=stop-1,
	                   d_traj=d_traj,
	                   temp_lb=min(df.temp),#296.15,
					   c1=params1[0][0],
					   c2=params1[0][1],
					   c3=params1[0][2],
					   c4=params1[0][3])
#	                   c1=0.001,
#	                   c2=0.6,
#	                   c3=1e-2,
#	                   c4=0.05)

actions = [0]
dists = [0]
states = []
state = model.reset()
states.append(state)
done = False
ind1 = 0
state, reward, done, info = model.step([h_traj[ind1]/100])
state, reward, done, info = model.step([0.5])
while not done:	
	    state, reward, done, info = model.step([h_traj[ind1]/100])
	    actions.append(h_traj[ind1])
	    # state, reward, done, info = model.step([0.5])
	    # actions.append(0.5)
	    dists.append(info['dist'])
	    states.append(state)
	    ind1 += 1

states = np.array(states)-273.15
t = df.time[0:len(states)]

plt.close('all')
fig, ax = plt.subplots(3, figsize=(10, 7))
ax[0].plot(t, actions, 'b--', linewidth=3)

ax[0].set_ylabel('PWM %')
ax[0].legend(['Heater'], loc='best')

ax[1].plot(t, states[:, 0], 'b-', linewidth=3, label=r'$T_c$')
ax[1].plot(t, states[:, 1], 'r--', linewidth=3, label=r'$T_h$')
#ax[1].plot(t,states[:,0], 'k-', linewidth=3, label="Model")
ax[1].set_ylabel(r'Temperature (C)')
ax[1].legend(loc='best')

ax[2].plot(t, dists, 'b-', linewidth=3, label=r'Fan',
           alpha=0.5)
ax[2].set_ylabel('PWM %')
ax[2].set_xlabel('Time (min)')
ax[2].legend(loc='best')

plt.figure(2)
plt.plot(t,states[:,0], 'r-', linewidth=3, label="Model")
plt.plot(t,df.temp[0:len(states)],'ko',label="Raw Data")
plt.ylabel(r'Temperature (C)')
plt.legend(loc='best')
plt.xlabel("Time (s)")
plt.show()
#
