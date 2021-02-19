import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "/data/pid_test(5).csv"


def visualizer(fp, cols=None):
    folder_path_txt = "../hidden/box_folder_path.txt"
    with open(folder_path_txt) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    box_folder_path = content[0]

    df = pd.read_csv(box_folder_path + fp)

    if cols is None:
        cols = df.columns[2:]

    fig, ax = plt.subplots(len(cols))
    for i in range(len(cols)):
        ax[i].plot(df['time'].values, df[cols[i]])
        ax[i].set_ylabel(cols[i])
        if i == len(cols) - 1:
            ax[i].set_xlabel('Time (s)')
    return fig, ax


visualizer(file_path)
