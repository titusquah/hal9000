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
col1_delta_means = []
col1_delta_medians = []
counter = 0

median_bar = 0.24
# median_bar = 0.085
# median_bar = 0.08
mean_bar = 2.1

test_cases = {}
test_cases_forecast = {}
plt.close('all')
while stop < len(rows):
    mini_dist = dists[start:stop]
    mini_forecast = forecasts[start:stop + 12 * 60]
    delta = np.abs(mini_dist[1:]-mini_dist[:-1])
    mean = np.mean(delta)
    median = np.median(delta)
    col1_delta_means.append(mean)
    col1_delta_medians.append(median)
    start = stop
    stop = start + 12 * 60
    if median > median_bar:

        test_cases['case{}'.format(counter+1)] = mini_dist
        test_cases_forecast['case{}'.format(counter+1)] = mini_forecast
        # if counter % 9 == 0:
        #     if counter != 0:
        #         plt.legend()
        #         plt.show()
        #     plt.figure()
        plt.figure()
        plt.plot(mini_dist, label=counter)
        plt.plot(mini_forecast, label="forecast")
        plt.legend()
        counter += 1
# plt.show()
col1_delta_means = np.array(col1_delta_means)
col1_delta_medians = np.array(col1_delta_medians)
test_df = pd.DataFrame(test_cases)
forecast_df = pd.DataFrame(test_cases_forecast)
print(len(list(test_df)))
# test_df.to_csv("dist_cases.csv")
