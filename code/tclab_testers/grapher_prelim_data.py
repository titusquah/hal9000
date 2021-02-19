import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 16}
matplotlib.rc('font', **font)


file_path = r"C:\Users\tq220\Box Sync\sync2020" \
            r"\Box Sync\hal9000_box_folder\data\test4_big_fan.csv"
df = pd.read_csv(file_path)
time = df['time'] / 60
fig, ax = plt.subplots(3, sharex=True, figsize=(10, 8))
stop = 10001
ax[0].plot(time[:stop], df['temp'].values[:stop], 'ro', markersize=4)
ax[0].set_ylabel('Temperature (°C)')
ax[0].grid()
ax[1].plot(time[:stop], df['heater_pwm'].values[:stop], 'b', linewidth=3)
ax[1].set_ylabel('Heater PWM (%)')
ax[1].grid()
ax[2].plot(time[:stop], df['fan_pwm'].values[:stop] * 100, 'b', linewidth=3)
ax[2].set_ylabel('Fan PWM (%)')
ax[2].set_xlabel('Time (min)')
ax[2].grid()
plt.tight_layout()
plt.savefig('prelim_data.png')
