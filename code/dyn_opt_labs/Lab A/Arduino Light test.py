# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:02:05 2019

@author: Titus
"""

import tclab
import time

# Connect to Arduino
a = tclab.TCLab()
print('LED On')
a.LED(100)
# Pause for 1 second
time.sleep(1.0)
print('LED Off')
a.LED(0)
a.close()