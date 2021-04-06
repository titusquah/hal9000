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
a1 = np.arange(6)
a2 = np.arange(3)
trials = np.array(list(itertools.product(*[a1, a2])))
np.random.shuffle(trials)
for trial in trials:
    if trial[0] == 0:
        forecast = 'nominal'
        forecast_scale_factor = 1
        file_path = "/data/real_{0}_test_case_{1}(7).csv".format(forecast,
                                                                 trial[1] + 1)
    elif trial[0] == 1:
        forecast = 'perfect'
        forecast_scale_factor = 1
        file_path = "/data/real_{0}_test_case_{1}(7).csv".format(forecast,
                                                                 trial[1] + 1)
    elif trial[0] == 2:
        forecast = tcm.get_forecast(int(trial[1]))
        forecast_scale_factor = 0
        file_path = r"/data/" \
                    r"real_{0}_scaled_" \
                    r"forecast_test_" \
                    r"case_{1}(7).csv".format(forecast_scale_factor,
                                              trial[1] + 1)
    elif trial[0] == 3:
        forecast = tcm.get_forecast(int(trial[1]))
        forecast_scale_factor = 1
        file_path = r"/data/" \
                    r"real_{0}_scaled_" \
                    r"forecast_test_" \
                    r"case_{1}(7).csv".format(forecast_scale_factor,
                                              trial[1] + 1)
    elif trial[0] == 4:
        forecast = tcm.get_forecast(int(trial[1]))
        forecast_scale_factor = 1.25
        file_path = r"/data/" \
                    r"real_{0}_scaled_" \
                    r"forecast_test_" \
                    r"case_{1}(7).csv".format(forecast_scale_factor,
                                              trial[1] + 1)
    else:
        forecast = tcm.get_forecast(int(trial[1]))
        forecast_scale_factor = 5
        file_path = r"/data/" \
                    r"real_{0}_scaled_" \
                    r"forecast_test_" \
                    r"case_{1}(7).csv".format(forecast_scale_factor,
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
    d_traj = tcm.get_d_traj(trial[1])
    try:
        tcm.general_mhe_mpc_test(dpin1,
                                 heater_board,
                                 tlb,
                                 d_traj,
                                 amb_temp,
                                 init_temp,
                                 penalty_scale=1e7,
                                 dmax=20,
                                 dcost=0.1,
                                 forecast=forecast,
                                 forecast_scale_factor=forecast_scale_factor,
                                 use_mhe=False,
                                 file_path=None,
                                 dt=1.5,
                                 look_back=31,
                                 look_forward=31,
                                 c1=0.001,
                                 c2=0.801088,
                                 c3=0.00388592,
                                 c4=0.09,
                                 )
    except:
        break
tcm.fan_cooling(dpin1, heater_board, temp_sp=None)
heater_board.close()
fan_board.exit()
