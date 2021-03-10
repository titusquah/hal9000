import numpy as np
import time
from tclab import TCLab
import pyfirmata
import pandas as pd
from gekko import GEKKO


# # Connect to Arduino
# heater_board = TCLab(port='4')
# fan_board = pyfirmata.Arduino("com5")
#
# it = pyfirmata.util.Iterator(fan_board)
# it.start()
#
# pntxt2 = "d:{}:o".format(3)
# dpin1 = fan_board.get_pin(pntxt2)
# dpin1.mode = 3
def get_d_traj(case, hold_time):
    folder_path_txt = "../hidden/box_folder_path.txt"
    with open(folder_path_txt) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    box_folder_path = content[0]
    file_path = "/data/dist_cases.csv"
    df = pd.read_csv(box_folder_path + file_path)
    d_traj = df['case{}'.format(case + 1)].values / 16 * 80 + 20
    d_traj = np.repeat(d_traj, hold_time)
    return d_traj


def fan_cooling(mini_dpin1, mini_heater_board, temp_sp=None):
    print("Starting cooling procedure")
    mini_heater_board.Q1(0)
    mini_heater_board.Q2(0)
    current_temp = mini_heater_board.T1
    mini_dpin1.write(1)
    start_time = time.time()
    prev_time = start_time
    sleep_max = 1
    times, temps, heater_pwms, fan_pwms = [], [], [], []
    if temp_sp:
        while current_temp > temp_sp - 1:
            sleep = sleep_max - (time.time() - prev_time)
            if sleep >= 0.01:
                time.sleep(sleep - 0.01)
            else:
                time.sleep(0.01)
            t = time.time()
            prev_time = t
            times.append(t - start_time)
            current_temp = mini_heater_board.T1
            temps.append(current_temp)
            heater_pwms.append(mini_heater_board.U1)
            if mini_dpin1.value:
                fan_pwms.append(mini_dpin1.value)
            else:
                fan_pwms.append(0)
            if len(temps) % 10 == 0:
                print("Current T = {0} °C".format(current_temp))
    else:
        stable = False
        steps_per_second = int(1 / sleep_max)
        hold_time = 10  # s
        back_index = int(steps_per_second * hold_time)
        tol = 0.3
        while not stable:
            sleep = sleep_max - (time.time() - prev_time)
            if sleep >= 0.01:
                time.sleep(sleep - 0.01)
            else:
                time.sleep(0.01)
            t = time.time()
            prev_time = t
            times.append(t - start_time)
            current_temp = mini_heater_board.T1
            temps.append(current_temp)
            heater_pwms.append(mini_heater_board.U1)
            if mini_dpin1.value:
                fan_pwms.append(mini_dpin1.value)
            else:
                fan_pwms.append(0)

            if len(times) > back_index:
                check_array = np.array(temps[-back_index:])
                max_diff = np.abs(np.max(check_array) - np.min(check_array))
                stable = max_diff < tol

            if len(temps) % 10 == 0:
                print("Current T = {0} °C".format(current_temp))

    mini_dpin1.write(0)
    print("Ending cooling procedure")
    print("Current T = {0} °C".format(current_temp))
    print("Current heater PWM = {0}".format(mini_heater_board.U1))
    print("Current fan PWM = {0}".format(mini_dpin1.value))
    return times, temps, heater_pwms, fan_pwms


def set_initial_temp(mini_dpin1,
                     mini_heater_board,
                     temp_sp,
                     tol,
                     hold_time,
                     file_path=None):
    print("Setting initial temperature to {0} °C".format(temp_sp))
    stable = False
    mini_dpin1.write(0)
    start_time = time.time()
    prev_time = start_time
    sleep_max = 1
    error = 0
    mv = 0
    dt = sleep_max
    steps_per_second = int(1 / sleep_max)
    times, temps, heater_pwms, fan_pwms = [], [], [], []
    current_temp = 0
    ind = 0
    while not stable:
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

        current_temp = mini_heater_board.T1
        temps.append(current_temp)
        old_error = error
        error = temp_sp - current_temp

        kc = 20  # 9.15*2
        ti = 70  # 312*0.25
        dmv = kc * (error - old_error + dt / ti * error)
        mv += dmv
        mv = np.clip(mv, 0, 100)
        mini_heater_board.Q1(mv)
        heater_pwms.append(mini_heater_board.U1)
        if mini_dpin1.value:
            fan_pwms.append(mini_dpin1.value)
        else:
            fan_pwms.append(0)
        temp_array = np.array(temps)
        errors = np.abs(temp_array - temp_sp)
        back_index = int(steps_per_second * hold_time)
        check_array = errors[-back_index:]
        stable = np.all(check_array < tol)
        if ind % 5 == 0 and file_path:
            df = pd.DataFrame({'time': times,
                               'temp': temps,
                               'heater_pwm': heater_pwms})
            df.to_csv(file_path)
        ind += 1
        if len(temps) % 10 == 0:
            print("Current T = {0} °C".format(current_temp))
    mini_heater_board.Q1(0)
    print("Ending set temp procedure")
    print("Current T = {0} °C".format(current_temp))
    print("Current heater PWM = {0}".format(mini_heater_board.U1))
    print("Current fan PWM = {0}".format(mini_dpin1.value))
    return times, temps, heater_pwms, fan_pwms


def nominal_mpc_test(mini_dpin1,
                     mini_heater_board,
                     temp_lb,
                     d_traj,
                     amb_temp,
                     init_temp,
                     file_path=None,
                     dt=1,
                     look_back=11,
                     look_forward=51,
                     ):
    max_change = 0.8
    min_change = 0.02
    decay_rate = 0.005
    mpc = GEKKO(name='tclab-mpc', remote=False, server='http://127.0.0.1')
    mhe = GEKKO(name='tclab-mpc', remote=False, server='http://127.0.0.1')
    mpc.time = np.linspace(0, (look_forward - 1) * dt, look_forward)
    mhe.time = np.linspace(0, (look_back - 1) * dt, look_back)
    apm_models = [mhe, mpc]
    for ind, apm_model in enumerate(apm_models):
        apm_model.c1 = apm_model.FV(value=0.39)
        apm_model.c2 = apm_model.FV(value=1.18)
        apm_model.c3 = apm_model.FV(value=0.26)
        apm_model.c4 = apm_model.FV(value=0.007)
        cs = [apm_model.c1, apm_model.c2, apm_model.c3, apm_model.c4]

        apm_model.fan_pwm = apm_model.MV(value=20)
        apm_model.fan_pwm.STATUS = 0
        apm_model.fan_pwm.FSTATUS = 1

        apm_model.heater_pwm = mpc.MV(value=100)
        apm_model.temp_heater = mpc.SV(value=init_temp)

        if ind == 0:
            for c in cs:
                c.STATUS = 0
                c.FSTATUS = 0
                c.LOWER = 0
                c.DMAX = max_change
            apm_model.heater_pwm.STATUS = 0
            apm_model.heater_pwm.FSTATUS = 1
            apm_model.temp_sensor = mhe.CV(value=init_temp, name='tc1')
            apm_model.temp_sensor.STATUS = 1
            apm_model.temp_sensor.FSTATUS = 1.
            apm_model.temp_sensor.MEAS_GAP = 0.1
        else:
            for c in cs:
                c.STATUS = 0
                c.FSTATUS = 1
            p = np.zeros(len(mpc.time))
            p[-1] = 1.0
            apm_model.final = mpc.Param(value=p)

            apm_model.heater_pwm.STATUS = 1
            apm_model.heater_pwm.FSTATUS = 0.
            # heater_pwm.DMAX = 20
            # heater_pwm.DCOST = 0.1
            apm_model.heater_pwm.LOWER = 0
            apm_model.heater_pwm.UPPER = 100

            apm_model.temp_sensor = mpc.SV(value=init_temp, name='tc1')
            apm_model.temp_sensor.STATUS = 0
            apm_model.temp_sensor.FSTATUS = 1.
        apm_model.h = mpc.Intermediate(apm_model.c1
                                       * apm_model.fan_pwm
                                       ** (apm_model.c2 - 1))
        apm_model.Equation(apm_model.temp_heater.dt()
                           == -apm_model.h * apm_model.temp_heater
                           + apm_model.c3 * apm_model.heater_pwm
                           + apm_model.c2 * apm_model.h * (
                                   amb_temp - apm_model.temp_heater)
                           * apm_model.fan_pwm)
        apm_model.Equation(
            (apm_model.temp_sensor.dt() == apm_model.c4
             * apm_model.temp_heater - apm_model.c4 * apm_model.temp_sensor))

        if ind == 0:
            

    print("Starting nominal MPC with T_lb =  {0} °C".format(temp_lb))

    mini_dpin1.write(0)
    start_time = time.time()
    prev_time = start_time
    sleep_max = 1
    dt = sleep_max
    steps_per_second = int(1 / sleep_max)
    times, temps, heater_pwms, fan_pwms = [], [], [], []
    current_temp = 0
    ind = 0
    while not stable:
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

        current_temp = mini_heater_board.T1
        temps.append(current_temp)
        old_error = error
        error = temp_sp - current_temp

        kc = 20  # 9.15*2
        ti = 70  # 312*0.25
        dmv = kc * (error - old_error + dt / ti * error)
        mv += dmv
        mv = np.clip(mv, 0, 100)
        mini_heater_board.Q1(mv)
        heater_pwms.append(mini_heater_board.U1)
        if mini_dpin1.value:
            fan_pwms.append(mini_dpin1.value)
        else:
            fan_pwms.append(0)
        temp_array = np.array(temps)
        errors = np.abs(temp_array - temp_sp)
        back_index = int(steps_per_second * hold_time)
        check_array = errors[-back_index:]
        stable = np.all(check_array < tol)
        if ind % 5 == 0 and file_path:
            df = pd.DataFrame({'time': times,
                               'temp': temps,
                               'heater_pwm': heater_pwms})
            df.to_csv(file_path)
        ind += 1
        if len(temps) % 10 == 0:
            print("Current T = {0} °C".format(current_temp))
    mini_heater_board.Q1(0)
    print("Ending set temp procedure")
    print("Current T = {0} °C".format(current_temp))
    print("Current heater PWM = {0}".format(mini_heater_board.U1))
    print("Current fan PWM = {0}".format(mini_dpin1.value))
    return times, temps, heater_pwms, fan_pwms
