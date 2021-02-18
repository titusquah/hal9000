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

temp_sp = 35
tol = 0.2
hold_time = 10
file_path = "/data/pid_test(1).csv"
folder_path_txt = "hidden/box_folder_path.txt"
with open(folder_path_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]
box_folder_path = content[0]
total_file_path = box_folder_path + file_path

times, temps, heater_pwms = set_initial_temp(heater_board,
                                             temp_sp,
                                             tol,
                                             hold_time,
                                             file_path=total_file_path)
heater_board.Q1(0)
heater_board.Q2(0)
dpin1.write(0)
print('Shutting down')
heater_board.close()
fan_board.exit()
