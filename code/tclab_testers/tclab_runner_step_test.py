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
trial = 1
if trial == 0:
    test = tcm.nominal_mpc_test
    test_name = 'nominal'
else:
    test = tcm.perfect_mpc_test
    test_name = 'perfect'
file_path = "/data/real_{0}_test_step(1).csv".format(test_name)
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
n_steps = 3600
step_20 = np.ones(60) * 20
step_100 = np.ones(60) * 100
hi = np.array([20])
mini_list = [hi]
counter = 1
ind1 = 0
while counter < n_steps:
    if ind1 % 2 == 0:
        mini_list.append(step_20)
    else:
        mini_list.append(step_100)
    ind1 += 1
    counter += 60
d_traj = np.concatenate(mini_list)
try:
    test(dpin1,
         heater_board,
         tlb,
         d_traj,
         amb_temp,
         init_temp,
         file_path=total_file_path,
         dt=1,
         look_back=31,
         look_forward=51,
         c1=0.00088341,
         c2=0.801088,
         c3=0.00388592,
         c4=1,
         )
except:
    pass
tcm.fan_cooling(dpin1, heater_board, temp_sp=None)
heater_board.close()
fan_board.exit()
