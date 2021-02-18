import numpy as np
import time
import matplotlib.pyplot as plt
from tclab import TCLab
import pyfirmata

# Connect to Arduino
heater_board = TCLab(port='4')
fan_board = pyfirmata.Arduino("com3")

it = pyfirmata.util.Iterator(fan_board)
it.start()

pntxt2 = "d:{}:o".format(3)
dpin1 = fan_board.get_pin(pntxt2)
dpin1.mode = 3