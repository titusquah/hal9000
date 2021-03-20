import numpy as np
import time
from tclab import TCLab
import pyfirmata
import pandas as pd
import tclab_modules as tcm
import matplotlib.pyplot as plt
import itertools

# Connect to Arduino
heater_board = TCLab(port='4')
fan_board = pyfirmata.Arduino("com5")

it = pyfirmata.util.Iterator(fan_board)
it.start()

pntxt2 = "d:{}:o".format(3)
dpin1 = fan_board.get_pin(pntxt2)
dpin1.mode = 3

tlb = 30  # °C
a1 = [0, 1]
a2 = np.arange(3)
trials = np.array(list(itertools.product(*[a1, a2])))
np.random.shuffle(trials)
for trial in trials:
    if trial[0] == 0:
        test = tcm.pid_test
        test_name = 'pid'
    else:
        test = tcm.ratio_ff_pid_test
        test_name = 'ff_pid'
    file_path = "/data/real_{0}_test_case_{1}(1).csv".format(test_name,
                                                             trial[1] + 1)
    folder_path_txt = "../hidden/box_folder_path.txt"
    with open(folder_path_txt) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    box_folder_path = content[0]
    total_file_path = box_folder_path + file_path

    temp_sp = None
    times1, temps1, heater_pwms1, fan_pwms1 = tcm.fan_cooling(dpin1,
                                                              heater_board,
                                                              temp_sp=None)

    amb_temp = min(temps1)

    temp_sp = tlb + 3
    tol = 0.2
    hold_time = 20
    times1, temps1, heater_pwms1, fan_pwms1 = tcm.set_initial_temp(dpin1,
                                                                   heater_board,
                                                                   temp_sp,
                                                                   tol,
                                                                   hold_time)
    init_temp = temps1[-1]
    dist_file_path = "/dist_cases(1)_w_time.csv"
    big_dist_df = pd.read_csv(box_folder_path+dist_file_path)
    dist_df = big_dist_df[['time', 'case{}'.format(trial[1]+1)]]
    dist_df.columns = ['time', 'fan_pwm']
    dt = 0.2
    try:
        test(dpin1,
             heater_board,
             tlb,
             amb_temp,
             dist_df,
             file_path=total_file_path,
             dt=dt,
             )
    except:
        break
tcm.fan_cooling(dpin1, heater_board, temp_sp=None)
heater_board.close()
fan_board.exit()
