from gekko import GEKKO
import numpy as np
import fan_tclab_gym as ftg
import matplotlib.pyplot as plt

c1 = 0.00073258
c2 = 0.800573
c3 = 0.00395524
c4 = 0.00284566
temp_lb = 310  # K

mpc = GEKKO(name='tclab-mpc')
mpc.time = np.arange(0, 5)



env = ftg.FanTempControlLabBlackBox(initial_temp=296.15,
                                    amb_temp=296.15,
                                    dt=0.1,
                                    max_time=6000,
                                    d_traj=None,
                                    temp_lb=temp_lb,
                                    c1=c1,
                                    c2=c2,
                                    c3=c3,
                                    c4=c4)
