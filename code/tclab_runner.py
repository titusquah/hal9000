import numpy as np
import time
from tclab import TCLab
import pyfirmata
import pandas as pd
from tclab_modules import fan_cooling, set_initial_temp
import matplotlib.pyplot as plt

# Connect to Arduino
heater_board = TCLab(port='4')
fan_board = pyfirmata.Arduino("com5")

it = pyfirmata.util.Iterator(fan_board)
it.start()

pntxt2 = "d:{}:o".format(3)
dpin1 = fan_board.get_pin(pntxt2)
dpin1.mode = 3

temp_sp = 50
tol = 0.1
hold_time = 30
file_path1 = "/data/pid_test(2).csv"
folder_path_txt = "hidden/box_folder_path.txt"
with open(folder_path_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]
box_folder_path = content[0]
total_file_path1 = box_folder_path + file_path1

times1, temps1, heater_pwms1, fan_pwms1 = set_initial_temp(dpin1,
                                             heater_board,
                                             temp_sp,
                                             tol,
                                             hold_time,
                                             file_path=total_file_path1)

temp_sp = 30
times2, temps2, heater_pwms2, fan_pwms2 = fan_cooling(dpin1,
                                             heater_board,
                                             temp_sp)
temp_sp = 60
file_path2 = "/data/pid_test(3).csv"
total_file_path2 = box_folder_path + file_path2
times3, temps3, heater_pwms3, fan_pwms3 = set_initial_temp(dpin1,
                                             heater_board,
                                             temp_sp,
                                             tol,
                                             hold_time,
                                             file_path=total_file_path2)
heater_board.Q1(0)
heater_board.Q2(0)
dpin1.write(0)
print('Shutting down')
heater_board.close()
fan_board.exit()

times, temps, heater_pwms, fan_pwms = [], [], [], []
times.extend(times1)
times.extend(times2)
times.extend(times3)
temps.extend(temps1)
temps.extend(temps2)
temps.extend(temps3)
heater_pwms.extend(heater_pwms1)
heater_pwms.extend(heater_pwms2)
heater_pwms.extend(heater_pwms3)
fan_pwms.extend(fan_pwms1)
fan_pwms.extend(fan_pwms2)
fan_pwms.extend(fan_pwms3)
df = pd.DataFrame({'time': times,
                   'temp': temps,
                   'heater_pwm': heater_pwms,
                   'fan_pwm': fan_pwms})
file_path3="/data/heating_cooling(1).csv"
total_file_path3 = box_folder_path + file_path3
df.to_csv(total_file_path3)





