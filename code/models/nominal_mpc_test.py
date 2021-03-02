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
initial_temp = 320
amb_temp = 296.15
log_barrier_tau = 0.5
penalty_scale = 10

env = ftg.FanTempControlLabBlackBox(initial_temp=initial_temp,
                                    amb_temp=amb_temp,
                                    dt=0.1,
                                    max_time=6000,
                                    d_traj=None,
                                    temp_lb=temp_lb,
                                    c1=c1,
                                    c2=c2,
                                    c3=c3,
                                    c4=c4)

mpc = GEKKO(name='tclab-mpc')
mpc.time = np.arange(0, 5)
c1 = mpc.FV(value=c1)
c2 = mpc.FV(value=c2)
c3 = mpc.FV(value=c3)
c4 = mpc.FV(value=c4)
log_barrier_tau = mpc.Param(value=log_barrier_tau)
temp_lb = mpc.Param(value=temp_lb)
penalty_scale = mpc.Param(value=penalty_scale)

fan_pwm = mpc.FV(value=c4)
fan_pwm.FSTATUS = 1

heater_pwm = mpc.MV(value=0)
heater_pwm.STATUS = 1
# Q1.DCOST = 0.1
# Q1.DMAX = 30
heater_pwm.LOWER = 0
heater_pwm.UPPER = 100

# State variables
temp_heater = mpc.SV(value=initial_temp)

# Measurements for model alignment
temp_sensor = mpc.CV(value=initial_temp, name='tc1')
temp_sensor.STATUS = 1  # minimize error between simulation and measurement
temp_sensor.FSTATUS = 1  # receive measurement
mpc.Equation(temp_heater.dt() == -c1 * fan_pwm ** (c2 - 1) * temp_heater
             + c3*heater_pwm
             + c1*c2*fan_pwm**(c2-1)*(amb_temp-temp_heater)*fan_pwm)
mpc.Equation((temp_sensor.dt() == c4*temp_heater-c4*temp_sensor))
mpc.Equation((temp_sensor.dt() == c4*temp_heater-c4*temp_sensor))
mpc.Obj(temp_heater)

