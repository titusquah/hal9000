import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

# folder_path_txt = "../hidden/box_folder_path.txt"
# with open(folder_path_txt) as f:
#     content = f.readlines()
# content = [x.strip() for x in content]
# box_folder_path = content[0]
# file_path = "/data/wind_disturbance_25.csv"
# with open(box_folder_path + file_path) as f:
#     content = f.readlines()
# content = [x.strip() for x in content]
# content.pop(0)

with open(
        r"C:\Users\tq220\Downloads"
        r"\Scenario-Forecasts-GAN-master"
        r"\Scenario-Forecasts-GAN-master\data\real.csv",
        'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
rows = np.array(rows, dtype=float)

with open(
        r"C:\Users\tq220\Box Sync\sync2020\Box "
        r"Sync\hal9000_box_folder\data\24_hour_ahead_full.csv",
        'r') as csvfile:
    reader = csv.reader(csvfile)
    rows1 = [row for row in reader]
rows1 = np.array(rows1, dtype=float)

delta = np.abs(rows[1:] - rows[0:-1])
delta_means = np.mean(delta, axis=0)
delta_medians = np.median(delta, axis=0)
# cols = [1, 24]
# plt.figure()
# for col in cols:
#
#     dists = rows[:, col]
#
#     start = 0
#     stop = start + 12 * 60 * 100
#     plt.plot(dists[start:stop], label=col)
# plt.legend()
# plt.plot()
start = 0
stop = start + 12 * 60
dists = rows[:, 0]
forecasts = rows1[:, 0]
len1 = len(dists)
len2 = len(forecasts)
len3 = min([len1, len2])
n_under = np.sum(forecasts[:len3] < dists[:len3])
print(n_under/len3)
alpha = 3
raised_forecast = alpha
