# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:24:21 2021

@author: kervi
"""

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
times, temps, heater_pwms, fan_pwms = [], [], [], []



temp_sp = None
times1, temps1, heater_pwms1, fan_pwms1 = fan_cooling(dpin1,
                                             heater_board,
                                             temp_sp)

heater_board.Q1(0)
heater_board.Q2(0)
dpin1.write(0)
print('Shutting down')
heater_board.close()
fan_board.exit()