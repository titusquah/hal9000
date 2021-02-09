import numpy as np
from tclab import TCLab
import time
import pandas as pd
import pyfirmata
import itertools
test_file_name = "test2.csv"
np.random.seed(int(time.time()))
tf = 20
nt = 600 * tf + 1

heater_board = TCLab(port='4')
fan_board = pyfirmata.Arduino("com3")

it = pyfirmata.util.Iterator(fan_board)
it.start()

pntxt2 = "d:{}:o".format(3)
dpin1 = fan_board.get_pin(pntxt2)
dpin1.mode = 3

start_time = time.time()
prev_time = start_time

times = []
temps = []
fan_pwms = []
heater_pwms = []

heater_pwm_values = [0, 25, 50, 75, 100]
fan_pwm_values = [0, 25, 50, 75, 100]
trials = list(itertools.product(*[heater_pwm_values, fan_pwm_values]))
np.random.shuffle(trials)
trials.insert(0, (0, 0))
time_schedule = [0, 5]
for i in range(len(trials) - 1):
    time_schedule.append(60 * 2.5 + time_schedule[-1])

heater_board.LED(100)

trial = 0
time_elapsed = 0
try:
    while time_elapsed < max(time_schedule):
        sleep_max = 0.1
        sleep = sleep_max - (time.time() - prev_time)
        if sleep >= 0.01:
            time.sleep(sleep - 0.01)
        else:
            time.sleep(0.01)

        t = time.time()
        dt = t - prev_time
        prev_time = t
        time_elapsed = t - start_time

        if time_elapsed >= time_schedule[trial + 1]:
            trial += 1
            df = pd.DataFrame({'time': times,
                               'temp': temps,
                               'heater_pwm': heater_pwms,
                               'fan_pwm': fan_pwms})
            df.to_csv(test_file_name)
        heater_pwm, fan_pwm = trials[trial]
        corrected_fan_pwm = fan_pwm / 100 * (1 - 0.2) + 0.2
        times.append(time_elapsed)
        temps.append(heater_board.T1)
        heater_pwms.append(heater_pwm)
        fan_pwms.append(corrected_fan_pwm)

        heater_board.Q1(heater_pwm)
        dpin1.write(corrected_fan_pwm)

    heater_board.Q1(0)
    heater_board.Q2(0)
    dpin1.write(0)
    heater_board.LED(0)
    heater_board.close()
    fan_board.exit()
    df = pd.DataFrame({'time': times,
                       'temp': temps,
                       'heater_pwm': heater_pwms,
                       'fan_pwm': fan_pwms})
    df.to_csv(test_file_name)
except KeyboardInterrupt:
    heater_board.Q1(0)
    heater_board.Q2(0)
    heater_board.LED(0)
    dpin1.write(0)
    print("KeyboardInterrupt shutdown")
    heater_board.close()
    fan_board.exit()
    df = pd.DataFrame({'time': times,
                       'temp': temps,
                       'heater_pwm': heater_pwms,
                       'fan_pwm': fan_pwms})
    df.to_csv(test_file_name)

except Exception as e:
    dpin1.write(0)
    heater_board.Q1(0)
    heater_board.Q2(0)
    heater_board.LED(0)
    print("{} shutdown".format(e))
    heater_board.close()
    fan_board.exit()
    df = pd.DataFrame({'time': times,
                       'temp': temps,
                       'heater_pwm': heater_pwms,
                       'fan_pwm': fan_pwms})
    df.to_csv(test_file_name)
    raise
