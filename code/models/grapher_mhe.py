import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 36}
matplotlib.rc('font', **font)
matplotlib.use('Qt5Agg')
start = 0
stop = 115965
folder_path_txt = "../hidden/box_folder_path.txt"
with open(folder_path_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]
box_folder_path = content[0]
file_path = '/data/mhe_test_{0}_{1}(4).csv'.format(start, stop)
df = pd.read_csv(box_folder_path + file_path)

mini_start = 0
mini_stop = 12001
fig, ax = plt.subplots(2, figsize=(19, 10))
ax[0].plot(df.time[mini_start:mini_stop],
           df.temp[mini_start:mini_stop],
           'bo', label=r'Measured $T_c$', markersize=8)
ax[0].plot(df.time[mini_start:mini_stop],
           df.est_temp[mini_start:mini_stop],
           'r-', label=r'Predicted $T_c$', linewidth=4)
ax[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1),
             fancybox=True, shadow=True, ncol=1)
ax[0].set_ylabel('$T_c$ (°C)')
ax[1].plot(df.time[mini_start:mini_stop],
           df.c1[mini_start:mini_stop], 'g-', label='$C_1$',
           linewidth=4)
ax[1].plot(df.time[mini_start:mini_stop],
           df.c2[mini_start:mini_stop], 'r-.', label='$C_2$',
           linewidth=4)
ax[1].plot(df.time[mini_start:mini_stop],
           df.c3[mini_start:mini_stop], 'b--', label='$C_3$',
           linewidth=4)
ax[1].plot(df.time[mini_start:mini_stop],
           df.c4[mini_start:mini_stop], 'k:', label='$C_4$',
           linewidth=4)
ax[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1),
             fancybox=True, shadow=True, ncol=2)
ax[1].set_ylabel('Parameter Value')
ax[1].set_xlabel('Time (s)')
plt.tight_layout()
# fig.savefig('mhe_fit5.eps', format='eps')
plt.show()

ss_resid = np.sum((df.temp[mini_start + 133:mini_stop].values
                   - df.est_temp[mini_start + 133:mini_stop].values) ** 2)
ss_total = np.sum((df.temp[mini_start + 133:mini_stop]
                   - df.temp[mini_start + 133:mini_stop].mean()) ** 2)
r_squared = 1 - (ss_resid / ss_total)
print(r_squared)
n = len(df.temp[mini_start + 133:mini_stop])
k = 4
adjusted_r_squared = 1 - (n - 1) / (n - (k + 1)) * (1 - r_squared)
print(adjusted_r_squared)
