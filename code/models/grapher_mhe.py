import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
mini_stop = stop
fig, ax = plt.subplots(2)
ax[0].plot(df.time[mini_start:mini_stop],
           df.temp[mini_start:mini_stop], 'bo', label='Measured', markersize=2)
ax[0].plot(df.time[mini_start:mini_stop],
           df.est_temp[mini_start:mini_stop], 'b-', label='Predicted')
ax[1].plot(df.time[mini_start:mini_stop], df.c2[mini_start:mini_stop], 'r-', label='$c_2$')
ax[1].plot(df.time[mini_start:mini_stop], df.c1[mini_start:mini_stop], 'g-', label='$c_1$')
ax[1].plot(df.time[mini_start:mini_stop], df.c3[mini_start:mini_stop], 'b-', label='$c_3$')
ax[1].plot(df.time[mini_start:mini_stop], df.c4[mini_start:mini_stop], 'k-', label='$c_4$')

plt.show()
