import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


start = 0
stop = 12001
folder_path_txt = "../hidden/box_folder_path.txt"
with open(folder_path_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]
box_folder_path = content[0]
file_path = '/data/mhe_test_{0}_{1}.csv'.format(start, stop)
df = pd.read_csv(box_folder_path + file_path)

plt.figure()
plt.plot(df.time, df.temp, 'bo', label='Measured', markersize=2)
plt.plot(df.time, df.est_temp, 'b-', label='Predicted')
plt.show()