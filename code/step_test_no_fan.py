import numpy as np
import time
from tclab import TCLab
import pandas as pd

folder_path_txt = "hidden/box_folder_path.txt"
with open(folder_path_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]
box_folder_path = content[0]
file_path = "/data/step_test_no_fan_50.csv"

# Connect to Arduino
heater_board = TCLab(port='4')

# Main Loop
start_time = time.time()
prev_time = start_time
times = []
temps = []

sleep_max = 1
steps_per_second = int(1 / sleep_max)

heater_pwms = np.concatenate((np.zeros(steps_per_second * 5),
                              np.ones(steps_per_second * 600),
                              np.zeros(steps_per_second * 300))) * 50
n = len(heater_pwms)
try:
    for i in range(1, n):
        # Sleep time
        sleep = sleep_max - (time.time() - prev_time)
        if sleep >= 0.01:
            time.sleep(sleep - 0.01)
        else:
            time.sleep(0.01)

        # Record time and change in time
        t = time.time()
        dt = t - prev_time
        prev_time = t
        times.append(t - start_time)

        # Read temperatures in Celsius
        temps.append(heater_board.T1)

        # Write new heater values (0-100)
        heater_board.Q1(heater_pwms[i])
        if i % 30 == 0:
            df = pd.DataFrame({'time': times,
                               'temp': temps,
                               'heater_pwm': heater_pwms[0:i]})
            df.to_csv(box_folder_path + file_path)

    # Turn off heaters
    heater_board.Q1(0)
    heater_board.Q2(0)
    heater_board.close()
    df = pd.DataFrame({'time': times,
                       'temp': temps,
                       'heater_pwm': heater_pwms[0:i]})
    df.to_csv(box_folder_path + file_path)

# Allow user to end loop with Ctrl-C
except KeyboardInterrupt:
    # Disconnect from Arduino
    heater_board.Q1(0)
    heater_board.Q2(0)
    print('Shutting down')
    heater_board.close()
    df = pd.DataFrame({'time': times,
                       'temp': temps,
                       'heater_pwm': heater_pwms[0:i]})
    df.to_csv(box_folder_path + file_path)


# Make sure serial connection still closes when there's an error
except:
    # Disconnect from Arduino
    heater_board.Q1(0)
    heater_board.Q2(0)
    print('Error: Shutting down')
    heater_board.close()
    df = pd.DataFrame({'time': times,
                       'temp': temps,
                       'heater_pwm': heater_pwms[0:i]})
    df.to_csv(box_folder_path + file_path)
    raise
