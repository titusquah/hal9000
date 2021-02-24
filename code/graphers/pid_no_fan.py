import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 16}
matplotlib.rc('font', **font)
matplotlib.use('Qt5Agg')


fp = "/data/heating_cooling(2).csv"
figsize = (10, 8)

folder_path_txt = "../hidden/box_folder_path.txt"
with open(folder_path_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]
box_folder_path = content[0]

df = pd.read_csv(box_folder_path + fp)
cols = df.columns[2:]
ylabels = ['Temperature (°C)', 'Heater PWM (%)', 'Fan PWM (%)']
temp_sps = []
mini_temp_sps = [df['temp'].min(), 40, 35, 50, df['temp'].values[-1]]
ind = 0
for ind1, val in enumerate(df['fan_pwm'].values):
    if ind1 == 0:
        temp_sps.append(mini_temp_sps[0])
    else:
        if val != df['fan_pwm'].values[ind1 - 1]:
            ind += 1
        temp_sps.append(mini_temp_sps[ind])

time = df['time'].values / 60
colors = ['r', 'b', 'b']
fig, ax = plt.subplots(len(cols), figsize=figsize)
for i in range(len(cols)):
    if i == 0:
        ax[i].plot(time, df[cols[i]], 'rx', label='Process data', markersize=2)
        ax[i].plot(time, temp_sps, 'r--', label='Set point')
        ax[i].legend(loc='best')
    else:
        ax[i].plot(time, df[cols[i]], "{}-".format(colors[i]))
    ax[i].set_ylabel(ylabels[i])
    if i == len(cols) - 1:
        ax[i].set_xlabel('Time (min)')
