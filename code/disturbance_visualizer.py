import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

folder_path_txt = "hidden/box_folder_path.txt"
with open(folder_path_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]
box_folder_path = content[0]
file_path = "/data/d_traj.csv"
df = pd.read_csv(box_folder_path + file_path)

start = 0
stop = 600
time = df['index'].values[start:stop]
dist = np.clip(pd.to_numeric(df['load'], errors='coerce').values[start:stop],
               0, None)

plt.figure()
plt.plot(time, dist)
plt.show()
