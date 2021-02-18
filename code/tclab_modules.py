import numpy as np
import time
from tclab import TCLab
import pyfirmata
import pandas as pd

# # Connect to Arduino
# heater_board = TCLab(port='4')
# fan_board = pyfirmata.Arduino("com5")
#
# it = pyfirmata.util.Iterator(fan_board)
# it.start()
#
# pntxt2 = "d:{}:o".format(3)
# dpin1 = fan_board.get_pin(pntxt2)
# dpin1.mode = 3


def fan_cooling(mini_dpin1, mini_heater_board, temp_sp):
    print("Starting cooling procedure")
    current_temp = mini_heater_board.T1
    mini_dpin1.write(1)
    while current_temp > temp_sp - 1:
        current_temp = mini_heater_board.T1
        time.sleep(1)
    print("Ending cooling procedure, current T = {0} °C".format(current_temp))
    mini_dpin1.write(0)
    return None


def set_initial_temp(mini_heater_board, temp_sp, tol, hold_time):
    print("Setting initial temperature to {0} °C".format(temp_sp))
    stable = False
    start_time = time.time()
    prev_time = start_time
    sleep_max = 1
    error = 0
    mv = 0
    dt = sleep_max
    steps_per_second = int(1 / sleep_max)
    times, temps, heater_pwms = [], [], []
    current_temp = 0
    while not stable:
        # Sleep time
        sleep = sleep_max - (time.time() - prev_time)
        if sleep >= 0.01:
            time.sleep(sleep - 0.01)
        else:
            time.sleep(0.01)

        # Record time and change in time
        t = time.time()
        dt = t - prev_time
        prev_time = t
        times.append(t - start_time)

        current_temp = mini_heater_board.T1
        temps.append(current_temp)
        old_error = error
        error = temp_sp - current_temp

        kc = 9.15
        ti = 312.2
        dmv = kc * (error - old_error + dt / ti * error)
        mv += dmv
        mv = np.clip(mv, 0, 100)
        mini_heater_board.Q1(mv)
        heater_pwms.append(mv)
        temp_array = np.array(temps)
        errors = np.abs(temp_array-temp_sp)
        back_index = int(steps_per_second*hold_time)
        check_array = errors[-back_index:]
        stable = np.all(check_array < tol)
    mini_heater_board.Q1(0)
    print("Ending set temp procedure, current T = {0} °C".format(current_temp))
    return times, temps, heater_pwms