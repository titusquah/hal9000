# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:15:03 2021

@author: Tony
"""


import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO

m = GEKKO(remote=False,server="http://127.0.0.1")

nt = 101
m.time = np.linspace(0,1,nt)

# Variables
x1 = m.Var(value=1)
x2 = m.Var(value=0)
u = m.Var(value=-0.75)

p = np.zeros(nt)
p[-1] = 1.0
final = m.Param(value=p)

# Equations
m.Equation(x1.dt()==u)
m.Equation(x2.dt()==x1**2 + u**2)

# Objective Function
m.Obj(x2*final)

m.options.IMODE = 6
m.solve()

plt.figure(1)
plt.plot(m.time,x1.value,'k:',LineWidth=2,label=r'$x_1$')
plt.plot(m.time,x2.value,'b-',LineWidth=2,label=r'$x_2$')
plt.plot(m.time,u.value,'r--',LineWidth=2,label=r'$u$')
plt.legend(loc='best')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()