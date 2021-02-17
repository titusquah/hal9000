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

# Make an MP4 animation?
make_mp4 = False
if make_mp4:
    import imageio  # required to make animation
    import os

    try:
        os.mkdir('./figures')
    except:
        pass

# Create plot
plt.figure(figsize=(12, 7))
plt.ion()
plt.show()

# Main Loop
start_time = time.time()
prev_time = start_time
times = []
temps = []

sleep_max = 0.5
steps_per_second = int(1 / sleep_max)

heater_pwms = np.ones(steps_per_second * 20) * 100
fan_pwms = np.concatenate((np.zeros(steps_per_second * 10),
                           np.ones(steps_per_second * 10))) * 100
n = len(heater_pwms)
try:
    for i in range(0, n):
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
        # Write new fan values (0-100)
        corrected_fan_pwm = fan_pwms[i] / 100 * (1 - 0.2) + 0.2
        dpin1.write(corrected_fan_pwm)

        # Plot
        plt.clf()
        ax = plt.subplot(3, 1, 1)
        ax.grid()
        plt.plot(times, temps, 'ro', label=r'$T_1$ measured')
        plt.ylabel('Temperature (°C)')
        plt.legend(loc=2)
        ax = plt.subplot(3, 1, 2)
        ax.grid()
        plt.plot(times, heater_pwms[0:i], 'k-', label='Heater PWM')
        plt.ylabel('%')
        plt.legend(loc='best')
        ax = plt.subplot(3, 1, 3)
        ax.grid()
        plt.plot(times, fan_pwms[0:i], 'r-', label=r'$Fan PWM$')
        plt.ylabel('%')
        plt.xlabel('Time (sec)')
        plt.legend(loc='best')
        plt.draw()
        plt.pause(0.05)
        if make_mp4:
            filename = './figures/plot_' + str(i + 10000) + '.png'
            plt.savefig(filename)

    # Turn off heaters
    heater_board.Q1(0)
    heater_board.Q2(0)
    dpin1.write(0)
    heater_board.close()
    fan_board.exit()

    # generate mp4 from png figures in batches of 350
    if make_mp4:
        images = []
        iset = 0
        for i in range(1, n):
            filename = './figures/plot_' + str(i + 10000) + '.png'
            images.append(imageio.imread(filename))
            if ((i + 1) % 350) == 0:
                imageio.mimsave('results_' + str(iset) + '.mp4', images)
                iset += 1
                images = []
        if images != []:
            imageio.mimsave('results_' + str(iset) + '.mp4', images)

# Allow user to end loop with Ctrl-C
except KeyboardInterrupt:
    # Disconnect from Arduino
    heater_board.Q1(0)
    heater_board.Q2(0)
    dpin1.write(0)
    print('Shutting down')
    heater_board.close()
    fan_board.exit()


# Make sure serial connection still closes when there's an error
except:
    # Disconnect from Arduino
    heater_board.Q1(0)
    heater_board.Q2(0)
    dpin1.write(0)
    print('Error: Shutting down')
    heater_board.close()
    fan_board.exit()
    raise
