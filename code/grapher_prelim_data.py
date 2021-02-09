import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 16}
matplotlib.rc('font', **font)

df = pd.read_csv('test2.csv')
time = df['time'] / 60
fig, ax = plt.subplots(3, sharex=True, figsize=(10, 8))
ax[0].plot(time, df['temp'], 'ro', markersize=4)
ax[0].set_ylabel('Temperature (°C)')
ax[0].grid()
ax[1].plot(time, df['heater_pwm'], 'b', linewidth=3)
ax[1].set_ylabel('Heater PWM (%)')
ax[1].grid()
ax[2].plot(time, df['fan_pwm'] * 100, 'b', linewidth=3)
ax[2].set_ylabel('Fan PWM (%)')
ax[2].set_xlabel('Time (min)')
ax[2].grid()
plt.tight_layout()
plt.savefig('prelim_data.png')
