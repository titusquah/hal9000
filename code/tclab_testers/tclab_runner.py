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

temp_sp = 40
tol = 0.2
hold_time = 30
file_path = "/data/pid_test(4).csv"
folder_path_txt = "hidden/box_folder_path.txt"
with open(folder_path_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]
box_folder_path = content[0]
total_file_path = box_folder_path + file_path
times, temps, heater_pwms, fan_pwms = [], [], [], []
times1, temps1, heater_pwms1, fan_pwms1 = set_initial_temp(dpin1,
                                             heater_board,
                                             temp_sp,
                                             tol,
                                             hold_time,
                                             file_path=total_file_path)

times.extend(times1)
temps.extend(temps1)
heater_pwms.extend(heater_pwms1)
fan_pwms.extend(fan_pwms1)


temp_sp = 35
times1, temps1, heater_pwms1, fan_pwms1 = fan_cooling(dpin1,
                                             heater_board,
                                             temp_sp)
times1 = np.array(times1)
times1 = times1 + times[-1]
times.extend(times1)
temps.extend(temps1)
heater_pwms.extend(heater_pwms1)
fan_pwms.extend(fan_pwms1)

temp_sp = 50
file_path = "/data/pid_test(5).csv"
total_file_path = box_folder_path + file_path
times1, temps1, heater_pwms1, fan_pwms1 = set_initial_temp(dpin1,
                                             heater_board,
                                             temp_sp,
                                             tol,
                                             hold_time,
                                             file_path=total_file_path)
times1 = np.array(times1)
times1 = times1 + times[-1]
times.extend(times1)
temps.extend(temps1)
heater_pwms.extend(heater_pwms1)
fan_pwms.extend(fan_pwms1)

temp_sp = None
times1, temps1, heater_pwms1, fan_pwms1 = fan_cooling(dpin1,
                                             heater_board,
                                             temp_sp)
times1 = np.array(times1)
times1 = times1 + times[-1]
times.extend(times1)
temps.extend(temps1)
heater_pwms.extend(heater_pwms1)
fan_pwms.extend(fan_pwms1)

heater_board.Q1(0)
heater_board.Q2(0)
dpin1.write(0)
print('Shutting down')
heater_board.close()
fan_board.exit()


df = pd.DataFrame({'time': times,
                   'temp': temps,
                   'heater_pwm': heater_pwms,
                   'fan_pwm': fan_pwms})
file_path="/data/heating_cooling(2).csv"
total_file_path3 = box_folder_path + file_path
df.to_csv(total_file_path)





