from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

folder_path_txt = "../hidden/box_folder_path.txt"
with open(folder_path_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]
box_folder_path = content[0]
file_path = "/data/test4_big_fan.csv"
df = pd.read_csv(box_folder_path + file_path)

c1 = 0.00464991
c2 = 0.801088
c3 = 0.0251691
c4 = 0.0184281

init_cs = [c1, c2, c3, c4]

dt = 0.155

start = 0
stop = 12001
lookback_time = 10  # s
save_file = box_folder_path + '/data/mhe_test_{0}_{1}(1).csv'.format(start, stop)

initial_temp = df['temp'][0] #+ 273.15
amb_temp = df['temp'][0] #+ 273.15

mhe = GEKKO(name='tclab-mhe', remote=False, server='http://127.0.0.1')
mhe.time = np.arange(0, lookback_time, dt)
c1 = mhe.FV(value=c1)
c2 = mhe.FV(value=c2)
c3 = mhe.FV(value=c3)
c4 = mhe.FV(value=c4)
cs = [c1, c2, c3, c4]

for c in cs:
    c.STATUS = 0
    c.FSTATUS = 0
    c.LOWER = 0
    c.DMAX = max(init_cs)

fan_pwm = mhe.MV(value=20)
fan_pwm.STATUS = 0
fan_pwm.FSTATUS = 1

heater_pwm = mhe.MV(value=100)
heater_pwm.STATUS = 0
heater_pwm.FSTATUS = 1

# State variables
temp_heater = mhe.SV(value=initial_temp)

# Measurements for model alignment
temp_sensor = mhe.CV(value=initial_temp, name='tc1')
temp_sensor.STATUS = 1  # minimize error between simulation and measurement
temp_sensor.FSTATUS = 1.  # receive measurement
temp_sensor.MEAS_GAP = 0.1

h = mhe.Intermediate(c1 * fan_pwm ** (c2 - 1))
mhe.Equation(temp_heater.dt() == -h * temp_heater
             + c3 * heater_pwm
             + c2 * h * (
                     amb_temp - temp_heater) * fan_pwm)
mhe.Equation((temp_sensor.dt() == c4 * temp_heater - c4 * temp_sensor))

mhe.options.IMODE = 5
mhe.options.EV_TYPE = 2
mhe.options.NODES = 2
mhe.options.SOLVER = 3
mhe.options.COLDSTART = 1
mhe.options.AUTO_COLD = 1
times = df['time'].values
heater_pwms = df['heater_pwm'].values
fan_pwms = df['fan_pwm'].values
temps = df['temp'].values

est_temps = []
est_cs = [[] for i in range(4)]
fail_counter = 0
for i in range(start, stop):
    if i - start > len(mhe.time):
        for ind1,c in enumerate(cs):
            if i % 4 == ind1:
                c.STATUS = 1
            else:
                c.STATUS = 0

    heater_pwm.MEAS = heater_pwms[i]
    fan_pwm.MEAS = fan_pwms[i]
    temp_sensor.MEAS = temps[i]

    try:
        mhe.solve(disp=False)
        oops = False
    except Exception:
        oops = True
        pass

    if oops:
        fail_counter += 1
        print("{} fails".format(fail_counter))
        if len(est_temps) != 0:
            est_temps.append(est_temps[-1])
            for ind1 in range(len(est_cs)):
                est_cs[ind1].append(est_cs[ind1][-1])
        else:
            est_temps.append(initial_temp)
            for ind1 in range(len(est_cs)):
                est_cs[ind1].append(init_cs[ind1])
    else:
        est_temps.append(temp_sensor.MODEL)
        for ind1 in range(len(est_cs)):
            est_cs[ind1].append(cs[ind1].NEWVAL)
    if i % 100 == 0:
        print(i)
        data_dict = {'time': times[start:i+1],
                     'temp': temps[start:i+1],
                     'est_temp': est_temps[start:i+1],
                     'fan_pwm': fan_pwms[start:i+1],
                     'heater_pwm': heater_pwms[start:i+1],
                     'c1': est_cs[0],
                     'c2': est_cs[1],
                     'c3': est_cs[2],
                     'c4': est_cs[3],
                     }
        data_df = pd.DataFrame(data_dict)

        data_df.to_csv(save_file, index=False)
data_dict = {'time': times[start:stop],
             'temp': temps[start:stop],
             'est_temp': est_temps[start:stop],
             'fan_pwm': fan_pwms[start:stop],
             'heater_pwm': heater_pwms[start:stop],
             'c1': est_cs[0],
             'c2': est_cs[1],
             'c3': est_cs[2],
             'c4': est_cs[3],
             }
data_df = pd.DataFrame(data_dict)

data_df.to_csv(save_file, index=False)
