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

tlb = 36  # °C
a1 = [1]
a2 = np.arange(7)
trials = np.array(list(itertools.product(*[a1, a2])))
np.random.shuffle(trials)
for trial in trials:
    if trial[0] == 0:
        test = tcm.nominal_mpc_test
        test_name = 'nominal'
    else:
        test = tcm.perfect_mpc_test
        test_name = 'perfect'
    file_path = "/data/real_{0}_test_case_{1}(2).csv".format(test_name,
                                                          trial[1] + 1)
    folder_path_txt = "../hidden/box_folder_path.txt"
    with open(folder_path_txt) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    box_folder_path = content[0]
    total_file_path = box_folder_path + file_path
    
#    try:
#        test(dpin1,
#             heater_board,
#             tlb,
#             d_traj,
#             amb_temp,
#             init_temp,
#             file_path=None,
#             dt=1,
#             look_back=11,
#             look_forward=51,
#             )
#    except:
#        break

    temp_sp = None
    times1, temps1, heater_pwms1, fan_pwms1 = tcm.fan_cooling(dpin1,
                                                              heater_board,
                                                              temp_sp=None)

    amb_temp = min(temps1)

    temp_sp = tlb + 1
    tol = 0.2
    hold_time = 20
    times1, temps1, heater_pwms1, fan_pwms1 = tcm.set_initial_temp(dpin1,
                                                                   heater_board,
                                                                   temp_sp,
                                                                   tol,
                                                                   hold_time)
    init_temp = temps1[-1]
    d_traj = tcm.get_d_traj(trial[1])
    total_file_path = None
    try:
        test(dpin1,
             heater_board,
             tlb,
             d_traj,
             amb_temp,
             init_temp,
             file_path=total_file_path,
             dt=1,
             look_back=11,
             look_forward=51,
             )
    except:
        break
tcm.fan_cooling(dpin1,heater_board,temp_sp=None)
heater_board.close()
fan_board.exit()
