import pandas as pd
import numpy as np


def get_d_traj(case, hold_time):
    folder_path_txt = "../hidden/box_folder_path.txt"
    with open(folder_path_txt) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    box_folder_path = content[0]
    file_path = "/data/dist_cases.csv"
    df = pd.read_csv(box_folder_path + file_path)
    d_traj = df['case{}'.format(case + 1)].values / 16 * 80 + 20
    d_traj = np.repeat(d_traj, hold_time)
    return d_traj
