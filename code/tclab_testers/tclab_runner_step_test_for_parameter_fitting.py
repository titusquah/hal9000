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

trials = [
    [[0, 100, 0], [0.2, 0.2, 0.2]],
    [[0, 100, 0], [0.4, 0.4, 0.4]],
    [[0, 100, 0], [0.6, 0.6, 0.6]],
    [[0, 100, 0], [0.8, 0.8, 0.8]],
    [[0, 100, 0], [1., 1., 1.]],
    [[100, 100, 100], [0.2, 0.4, 0.2]],
    [[100, 100, 100], [0.2, 0.6, 0.2]],
    [[100, 100, 100], [0.2, 0.8, 0.2]],
    [[100, 100, 100], [0.2, 1., 0.2]],
]
for trial in trials:
    file_path = "/data/heater_{0}_{1}_fan_{2}_3}.csv".format(trial[0][0],
                                                             trial[0][1],
                                                             trial[1][0],
                                                             trial[1][1], )
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

    try:
        tcm.step_tester(dpin1,
                        heater_board,
                        amb_temp,
                        tol=0.4,
                        hold_time=10,
                        fan_pwms_order=trial[1],
                        heater_pwms_order=trial[0],
                        file_path=total_file_path)
    except:
        break
tcm.fan_cooling(dpin1, heater_board, temp_sp=None)
heater_board.close()
fan_board.exit()
