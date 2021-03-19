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
def get_d_traj(case, hold_time=5):
    folder_path_txt = "../hidden/box_folder_path.txt"
    with open(folder_path_txt) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    box_folder_path = content[0]
    file_path = "/data/dist_cases(1).csv"
    df = pd.read_csv(box_folder_path + file_path)
    d_traj = df['case{}'.format(case + 1)].values / 16 * 80 + 20
    d_traj = np.repeat(d_traj, hold_time)
    return d_traj


def fan_cooling(mini_dpin1,
                mini_heater_board,
                temp_sp=None,
                hold_time=20,
                tol=0.3):
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
        back_index = int(steps_per_second * hold_time)
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
                     look_back=31,
                     look_forward=51,
                     c1=0.00088341,
                     c2=0.801088,
                     c3=0.00388592,
                     c4=0.09,
                     ):
    max_change = 0.8
    min_change = 0.02
    decay_rate = 0.25
    penalty_scale = 1e5
    steepness = 10
    fv_update_rate = 5  # s
    init_cs = [c1, c2, c3, c4]
    rel_max_change = 0.1
    mpc = GEKKO(name='tclab-mpc', remote=False, server='http://127.0.0.1')
    mhe = GEKKO(name='tclab-mhe', remote=False, server='http://127.0.0.1')
    mpc.time = np.linspace(0, (look_forward - 1) * dt, look_forward)
    mhe.time = np.linspace(0, (look_back - 1) * dt, look_back)
    apm_models = [mhe, mpc]
    for ind, apm_model in enumerate(apm_models):
        apm_model.c1 = apm_model.FV(value=c1)
        apm_model.c2 = apm_model.FV(value=c2)
        apm_model.c3 = apm_model.FV(value=c3)
        apm_model.c4 = apm_model.FV(value=c4)
        cs = [apm_model.c1, apm_model.c2, apm_model.c3, apm_model.c4]

        apm_model.heater_pwm = apm_model.MV(value=0)
        apm_model.temp_heater = apm_model.SV(value=init_temp)

        if ind == 0:
            apm_model.fan_pwm = apm_model.MV(value=20)
            apm_model.fan_pwm.STATUS = 0
            apm_model.fan_pwm.FSTATUS = 1
            for ind1, c in enumerate(cs):
                c.STATUS = 0
                c.FSTATUS = 0
                c.LOWER = 1e-4
                c.UPPER = 2
                c.DMAX = max_change
            apm_model.heater_pwm.STATUS = 0
            apm_model.heater_pwm.FSTATUS = 1
            apm_model.temp_sensor = apm_model.CV(value=init_temp, name='tc1')
            apm_model.temp_sensor.STATUS = 1
            apm_model.temp_sensor.FSTATUS = 1.
            apm_model.temp_sensor.MEAS_GAP = 0.1
        else:
            apm_model.fan_pwm = apm_model.FV(value=20)
            apm_model.fan_pwm.STATUS = 0
            apm_model.fan_pwm.FSTATUS = 1
            for c in cs:
                c.STATUS = 0
                c.FSTATUS = 1
            p = np.zeros(len(apm_model.time))
            p[-1] = 1.0
            apm_model.final = apm_model.Param(value=p)

            apm_model.heater_pwm.STATUS = 1
            apm_model.heater_pwm.FSTATUS = 0.
            apm_model.heater_pwm.DMAX = 20
            apm_model.heater_pwm.DCOST = 0.5
            apm_model.heater_pwm.LOWER = 0
            apm_model.heater_pwm.UPPER = 100

            apm_model.temp_sensor = apm_model.SV(value=init_temp, name='tc1')
            apm_model.temp_sensor.FSTATUS = 1.
        apm_model.h = apm_model.Intermediate(apm_model.c1
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
            apm_model.options.IMODE = 5
            apm_model.EV_TYPE = 1
        else:
            apm_model.Obj(
                apm_model.integral(
                    apm_model.heater_pwm + penalty_scale * apm_model.log(
                        1 + apm_model.exp(steepness
                                          * (temp_lb
                                             - apm_model.temp_sensor)))
                    / steepness) * apm_model.final)
            apm_model.options.IMODE = 6
        apm_model.options.NODES = 2
        apm_model.options.SOLVER = 3
        apm_model.options.COLDSTART = 1
        apm_model.options.AUTO_COLD = 1

    print("Starting nominal MPC with T_lb =  {0} °C".format(temp_lb))

    mini_dpin1.write(0)
    mini_heater_board.Q1(0)
    start_time = time.time()
    prev_time = start_time
    sleep_max = dt
    steps_per_second = int(1 / sleep_max)
    times, temps, heater_pwms, fan_pwms = [], [], [], []
    est_temps = []
    c1s, c2s, c3s, c4s = [], [], [], []
    current_temp = 0
    update_counter = 0
    ind = 0
    mhe.temp_sensor.MEAS = mini_heater_board.T1
    mpc.temp_sensor.MEAS = mini_heater_board.T1
    for ind1, dist in enumerate(d_traj):
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

        mini_dpin1.write(dist / 100)

        current_temp = mini_heater_board.T1
        current_dist = mini_dpin1.value

        mhe_cs = [mhe.c1, mhe.c3]
        if (ind1 % fv_update_rate == 0
                and ind1 > look_back):
            for ind2, mhe_c in enumerate(mhe_cs):
                mhe_c.STATUS = 1
                #                mhe_c.STATUS = 0
                update_counter += 1
                mhe_c.DMAX = max_change * np.exp(
                    -decay_rate * update_counter) + min_change
        else:
            for ind2, mhe_c in enumerate(mhe_cs):
                mhe_c.STATUS = 0
        mhe.heater_pwm.MEAS = mini_heater_board.U1
        mhe.fan_pwm.MEAS = current_dist * 100
        mhe.temp_sensor.MEAS = current_temp
        try:
            mhe.solve(disp=False)
            oops = False
        except Exception:
            oops = True
            pass

        est_temps.append(mhe.temp_sensor.MODEL)

        if oops:
            if ind1 != 0:

                c1s.append(c1s[-1])
                c2s.append(c2s[-1])
                c3s.append(c3s[-1])
                c4s.append(c4s[-1])
            else:

                c1s.append(init_cs[0])
                c2s.append(init_cs[1])
                c3s.append(init_cs[2])
                c4s.append(init_cs[3])
        else:
            c1s.append(mhe.c1.NEWVAL)
            c2s.append(mhe.c2.NEWVAL)
            c3s.append(mhe.c3.NEWVAL)
            c4s.append(mhe.c4.NEWVAL)

        mpc.temp_sensor.MEAS = current_temp
        mpc.fan_pwm.MEAS = current_dist * 100
        mpc.c1.MEAS = c1s[-1]
        mpc.c2.MEAS = c2s[-1]
        mpc.c3.MEAS = c3s[-1]
        mpc.c4.MEAS = c4s[-1]
        try:
            mpc.solve(disp=False)
            if mpc.options.APPSTATUS == 1:
                # Retrieve new values
                action = mpc.heater_pwm.NEWVAL / 100
            #            print(heater_pwm.VALUE)
            else:
                action = 1
        except Exception as e:
            action = 1

        mini_heater_board.Q1(action * 100)
        temps.append(current_temp)
        heater_pwms.append(mini_heater_board.U1)
        fan_pwms.append(current_dist)
        if file_path:
            if ind1 % 10 == 0:
                df = pd.DataFrame({'time': times,
                                   'temp': temps,
                                   'temp_lb': temp_lb * np.ones(len(times)),
                                   'est_temp': est_temps,
                                   'heater_pwm': heater_pwms,
                                   'fan_pwm': fan_pwms,
                                   'c1': c1s,
                                   'c2': c2s,
                                   'c3': c3s,
                                   'c4': c4s})
                df.to_csv(file_path)
            elif ind1 == len(d_traj) - 1:
                df = pd.DataFrame({'time': times,
                                   'temp': temps,
                                   'temp_lb': temp_lb * np.ones(len(times)),
                                   'est_temp': est_temps,
                                   'heater_pwm': heater_pwms,
                                   'fan_pwm': fan_pwms,
                                   'c1': c1s,
                                   'c2': c2s,
                                   'c3': c3s,
                                   'c4': c4s})
                df.to_csv(file_path)
    mini_dpin1.write(0)
    mini_heater_board.Q1(0)
    print("Ending Nominal MPC test")
    print("Current T = {0} °C".format(current_temp))
    print("Current heater PWM = {0}".format(mini_heater_board.U1))
    print("Current fan PWM = {0}".format(mini_dpin1.value))
    return times, temps, heater_pwms, fan_pwms, c1s, c2s, c3s, c4s


def perfect_mpc_test(mini_dpin1,
                     mini_heater_board,
                     temp_lb,
                     d_traj,
                     amb_temp,
                     init_temp,
                     file_path=None,
                     dt=1,
                     look_back=31,
                     look_forward=51,
                     c1=0.00088341,
                     c2=0.801088,
                     c3=0.00388592,
                     c4=0.09,
                     ):
    max_change = 0.8
    min_change = 0.02
    decay_rate = 0.25
    fv_update_rate = 5  # s
    rel_max_change = 0.1
    penalty_scale = 1e5
    steepness = 10
    init_cs = [c1, c2, c3, c4]

    d_traj_extend = np.concatenate([d_traj, d_traj])

    mpc = GEKKO(name='tclab-mpc', remote=False, server='http://127.0.0.1')
    mhe = GEKKO(name='tclab-mhe', remote=False, server='http://127.0.0.1')
    mpc.time = np.linspace(0, (look_forward - 1) * dt, look_forward)
    mhe.time = np.linspace(0, (look_back - 1) * dt, look_back)
    apm_models = [mhe, mpc]
    for ind, apm_model in enumerate(apm_models):
        apm_model.c1 = apm_model.FV(value=c1)
        apm_model.c2 = apm_model.FV(value=c2)
        apm_model.c3 = apm_model.FV(value=c3)
        apm_model.c4 = apm_model.FV(value=c4)
        cs = [apm_model.c1, apm_model.c2, apm_model.c3, apm_model.c4]

        apm_model.fan_pwm = apm_model.FV(value=20)
        apm_model.fan_pwm.STATUS = 0
        apm_model.fan_pwm.FSTATUS = 1

        apm_model.heater_pwm = apm_model.MV(value=0)
        apm_model.temp_heater = apm_model.SV(value=init_temp)

        if ind == 0:
            apm_model.fan_pwm = apm_model.MV(value=20)
            apm_model.fan_pwm.STATUS = 0
            apm_model.fan_pwm.FSTATUS = 1
            for ind1, c in enumerate(cs):
                c.STATUS = 0
                c.FSTATUS = 0
                c.LOWER = 0
                c.DMAX = rel_max_change * init_cs[ind1]
            apm_model.heater_pwm.STATUS = 0
            apm_model.heater_pwm.FSTATUS = 1
            apm_model.temp_sensor = apm_model.CV(value=init_temp,
                                                 name='mhe_tc1')
            apm_model.temp_sensor.STATUS = 1
            apm_model.temp_sensor.FSTATUS = 1.
            apm_model.temp_sensor.MEAS_GAP = 0.1
        else:
            apm_model.fan_pwm = apm_model.FV(value=20)
            apm_model.fan_pwm.STATUS = 0
            apm_model.fan_pwm.FSTATUS = 1
            for c in cs:
                c.STATUS = 0
                c.FSTATUS = 1
            p = np.zeros(len(apm_model.time))
            p[-1] = 1.0
            apm_model.final = apm_model.Param(value=p)

            apm_model.heater_pwm.STATUS = 1
            apm_model.heater_pwm.FSTATUS = 0.
            apm_model.heater_pwm.DMAX = 20
            apm_model.heater_pwm.DCOST = 0.5
            apm_model.heater_pwm.LOWER = 0
            apm_model.heater_pwm.UPPER = 100

            apm_model.temp_sensor = apm_model.SV(value=init_temp,
                                                 name='mpc_tc1')
            apm_model.temp_sensor.FSTATUS = 1.
        apm_model.h = apm_model.Intermediate(apm_model.c1
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
            apm_model.options.IMODE = 5
            apm_model.EV_TYPE = 1
        else:
            apm_model.Obj(
                apm_model.integral(
                    apm_model.heater_pwm + penalty_scale * apm_model.log(
                        1 + apm_model.exp(steepness
                                          * (temp_lb
                                             - apm_model.temp_sensor)))
                    / steepness) * apm_model.final)
            apm_model.options.IMODE = 6
        apm_model.options.NODES = 2
        apm_model.options.SOLVER = 3
        apm_model.options.COLDSTART = 1
        apm_model.options.AUTO_COLD = 1

    print("Starting Perfect MPC with T_lb =  {0} °C".format(temp_lb))

    mini_dpin1.write(0)
    mini_heater_board.Q1(0)
    start_time = time.time()
    prev_time = start_time
    sleep_max = dt
    steps_per_second = int(1 / sleep_max)
    times, temps, heater_pwms, fan_pwms = [], [], [], []
    est_temps = []
    c1s, c2s, c3s, c4s = [], [], [], []
    current_temp = 0
    update_counter = 0
    ind = 0
    mhe.temp_sensor.VALUE = mini_heater_board.T1
    mpc.temp_sensor.VALUE = mini_heater_board.T1
    for ind1, dist in enumerate(d_traj):
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

        mini_dpin1.write(dist / 100)

        current_temp = mini_heater_board.T1
        current_dist = mini_dpin1.value

        mhe_cs = [mhe.c1, mhe.c3]
        if (ind1 % fv_update_rate == 0
                and ind1 > look_back):
            for ind2, mhe_c in enumerate(mhe_cs):
                mhe_c.STATUS = 1
                #                mhe_c.STATUS = 0
                update_counter += 1
                mhe_c.DMAX = max_change * np.exp(
                    -decay_rate * update_counter) + min_change
        else:
            for ind2, mhe_c in enumerate(mhe_cs):
                mhe_c.STATUS = 0
        mhe.heater_pwm.MEAS = mini_heater_board.U1
        mhe.fan_pwm.MEAS = current_dist * 100
        mhe.temp_sensor.MEAS = current_temp
        try:
            mhe.solve(disp=False)
            oops = False
        except Exception:
            oops = True
            pass
        est_temps.append(mhe.temp_sensor.MODEL)
        if oops:
            if ind1 != 0:
                c1s.append(c1s[-1])
                c2s.append(c2s[-1])
                c3s.append(c3s[-1])
                c4s.append(c4s[-1])
            else:
                c1s.append(init_cs[0])
                c2s.append(init_cs[1])
                c3s.append(init_cs[2])
                c4s.append(init_cs[3])
        else:
            c1s.append(mhe.c1.NEWVAL)
            c2s.append(mhe.c2.NEWVAL)
            c3s.append(mhe.c3.NEWVAL)
            c4s.append(mhe.c4.NEWVAL)

        mpc.temp_sensor.MEAS = current_temp
        mpc.fan_pwm.VALUE = d_traj_extend[ind1:ind1 + look_forward]
        mpc.c1.MEAS = c1s[-1]
        mpc.c2.MEAS = c2s[-1]
        mpc.c3.MEAS = c3s[-1]
        mpc.c4.MEAS = c4s[-1]
        try:
            mpc.solve(disp=False)
            if mpc.options.APPSTATUS == 1:
                # Retrieve new values
                action = mpc.heater_pwm.NEWVAL / 100
            #            print(heater_pwm.VALUE)
            else:
                action = 1
        except Exception as e:
            action = 1

        mini_heater_board.Q1(action * 100)
        temps.append(current_temp)
        heater_pwms.append(mini_heater_board.U1)
        fan_pwms.append(current_dist)
        if file_path:
            if ind1 % 10 == 0:
                df = pd.DataFrame({'time': times,
                                   'temp': temps,
                                   'temp_lb': temp_lb * np.ones(len(times)),
                                   'est_temp': est_temps,
                                   'heater_pwm': heater_pwms,
                                   'fan_pwm': fan_pwms,
                                   'c1': c1s,
                                   'c2': c2s,
                                   'c3': c3s,
                                   'c4': c4s})
                df.to_csv(file_path)
            elif ind1 == len(d_traj) - 1:
                df = pd.DataFrame({'time': times,
                                   'temp': temps,
                                   'temp_lb': temp_lb * np.ones(len(times)),
                                   'est_temp': est_temps,
                                   'heater_pwm': heater_pwms,
                                   'fan_pwm': fan_pwms,
                                   'c1': c1s,
                                   'c2': c2s,
                                   'c3': c3s,
                                   'c4': c4s})
                df.to_csv(file_path)
    mini_dpin1.write(0)
    mini_heater_board.Q1(0)
    print("Ending Perfect MPC test")
    print("Current T = {0} °C".format(current_temp))
    print("Current heater PWM = {0}".format(mini_heater_board.U1))
    print("Current fan PWM = {0}".format(mini_dpin1.value))
    return times, temps, heater_pwms, fan_pwms, c1s, c2s, c3s, c4s


def step_tester(mini_dpin1,
                mini_heater_board,
                amb_temp,
                tol,
                hold_time,
                fan_pwms_order=None,
                heater_pwms_order=None,
                file_path=None):
    if fan_pwms_order is None:
        fan_pwms_order = [0.2, 0.2, 0.2]
    if heater_pwms_order is None:
        heater_pwms_order = [0, 100, 0]
    start_time = time.time()
    prev_time = start_time
    sleep_max = 1
    steps_per_second = int(1 / sleep_max)
    times, temps, heater_pwms, fan_pwms = [], [], [], []
    current_temp = 0
    for ind1 in range(len(fan_pwms_order)):
        ind = 0
        stable = False
        mini_dpin1.write(fan_pwms_order[ind1])
        mini_heater_board.Q1(heater_pwms_order[ind1])
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

            heater_pwms.append(mini_heater_board.U1)
            if mini_dpin1.value:
                fan_pwms.append(mini_dpin1.value)
            else:
                fan_pwms.append(0)
            temp_array = np.array(temps)
            if len(temp_array) > hold_time + 5 and ind > hold_time * 2:
                diffs = np.abs(temp_array[1:] - temp_array[:-1])
                back_index = int(steps_per_second * hold_time)
                check_array = temp_array[-back_index:]
                max_diff = np.max(check_array) - np.min(check_array)
                stable = max_diff < tol
            if ind % 5 == 0 and file_path:
                df = pd.DataFrame({'time': times,
                                   'temp': temps,
                                   'amb_temp': amb_temp * np.ones(len(times)),
                                   'heater_pwm': heater_pwms,
                                   'fan_pwm': fan_pwms})
                df.to_csv(file_path)
            ind += 1
            if len(temps) % 10 == 0:
                print("Current T = {0} °C".format(current_temp))
    df = pd.DataFrame({'time': times,
                       'temp': temps,
                       'amb_temp': amb_temp * np.ones(len(times)),
                       'heater_pwm': heater_pwms,
                       'fan_pwm': fan_pwms})
    df.to_csv(file_path)
    mini_dpin1.write(0)
    mini_heater_board.Q1(0)
    print("Ending set temp procedure")
    print("Current T = {0} °C".format(current_temp))
    print("Current heater PWM = {0}".format(mini_heater_board.U1))
    print("Current fan PWM = {0}".format(mini_dpin1.value))
    return times, temps, heater_pwms, fan_pwms


def pid_tuning(mini_dpin1,
               mini_heater_board,
               temp_sp,
               amb_temp,
               dist,
               tol,
               dt,
               hold_time,
               kc=20,
               ti=70,
               file_path=None):
    print("Setting temperature to {0} °C".format(temp_sp))
    stable = False
    start_time = time.time()
    prev_time = start_time
    sleep_max = dt
    error = 0
    mv = 0
    steps_per_second = int(1 / sleep_max)
    times, temps, heater_pwms, fan_pwms = [], [], [], []
    current_temp = 0
    ind = 0
    mini_dpin1.write(dist)
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
                               'temp_lb': temp_sp*np.ones(len(times)),
                               'amb_temp': amb_temp*np.ones(len(times)),
                               'fan_pwm': fan_pwms,
                               'heater_pwm': heater_pwms})
            df.to_csv(file_path)
        ind += 1
        if len(temps) % 10 == 0:
            print("Current T = {0} °C".format(current_temp))
    mini_heater_board.Q1(0)
    mini_dpin1.write(0)
    print("Ending set temp procedure")
    print("Current T = {0} °C".format(current_temp))
    print("Current heater PWM = {0}".format(mini_heater_board.U1))
    print("Current fan PWM = {0}".format(mini_dpin1.value))
    return times, temps, heater_pwms, fan_pwms


# def feedforward_pid_test(mini_dpin1,
#                          mini_heater_board,
#                          temp_lb,
#                          d_traj,
#                          file_path=None,
#                          dt=1, ):
#     x1 = [6159.780193306501, 54075.10058481033, 648.4392669781367]
#     x = [0.028363165980579814, 339.5272119702745, 219.29165427892744]
#
#     ####### FEEDFORWARD CODE ########
#
#     kff = -x[0] / x1[0]
#     theta_ff = x[2] - x1[2]
#     lead = x[1]
#     lag = x1[1]
#
#     tf = 1000  # final time
#     n = int(tf + 1)  # number of time points
#
#     # time span for the simulation, cycle every 1 sec
#     ts = np.linspace(0, tf, n)
#     delta_t = ts[1] - ts[0]
#
#     # disturbances
#     DP = np.zeros(n)
#     Fout = np.ones(n) * 2.0
#     Fin = np.zeros(n)
#
#     # Desired level (set point)
#     SP = 33
#     # level initial condition
#     Level0 = SP
#
#     # initial valve position
#     valve = 0
#     # Controller bias
#     ubias = valve
#     # valve opening (0-100%)
#     u1 = np.ones(n) * valve
#
#     # for storing the results
#     P = np.zeros(n)  # proportional
#     I = np.zeros(n)  # integral
#     ie = np.zeros(n)
#
#     # Controller tuning
#     Kc = x1[0]
#     tauI = x1[1]
#
#     # Connect to Arduino
#     heater_board = TCLab(port='4')
#
#     # Connect to Fan
#     board = pyfirmata.Arduino(
#         "com5")
#
#     it = pyfirmata.util.Iterator(board)
#     it.start()
#     # Main Loop
#     start_time = time.time()
#     prev_time = start_time
#     times = []
#     temps = []
#
#     sleep_max = dt
#     steps_per_second = int(1 / sleep_max)
#
#     d_traj = get_d_traj(0, 5)
#     pntxt2 = "d:{}:o".format(3)
#     dpin = board.get_pin(pntxt2)
#     dpin.mode = 3
#     j = 0
#
#     try:
#         for i in range(1, n):
#             # Sleep time
#             sleep = sleep_max - (time.time() - prev_time)
#             if sleep >= 0.01:
#                 time.sleep(sleep - 0.01)
#             else:
#                 time.sleep(0.01)
#
#             # Record time and change in time
#             t = time.time()
#             dt = t - prev_time
#             prev_time = t
#             times.append(t - start_time)
#
#             # Read temperatures in Celsius
#             temps.append(heater_board.T1)
#
#             # Write new heater values (0-100)
#             # pntxt2 = "d:{}:o".format(3)
#             #        dpin = board.get_pin(fan_pwms[i])
#             #        dpin.mode = 3
#
#             # inlet pressure (bar) disturbance
#             if i & tf == 0:
#
#                 DP[i] = data2[case[zz]][j]
#                 j += 1
#             else:
#                 DP[i] = data2[case[zz]][j]
#
#             # inlet mass flow
#             Fin[i] = DP[i]
#
#             # outlet flow (kg/sec) disturbance (change every 10 seconds)
#             if np.mod(i + 1, 500) == 100:
#                 Fout[i] = Fout[i - 1] + 10.0
#             elif np.mod(i + 1, 500) == 350:
#                 Fout[i] = Fout[i - 1] - 10.0
#             else:
#                 if i >= 1:
#                     Fout[i] = Fout[i - 1]
#             dpin.write(Fout[i])
#             # PI controller
#             # calculate the error
#             error = SP - Level0
#             P[i] = Kc * error
#             if i >= 1:  # calculate starting on second cycle
#                 ie[i] = ie[i - 1] + error * delta_t
#                 I[i] = (Kc / tauI) * ie[i]
#             valve = ubias + P[i] + I[i] + kff * (Fout[i] - Fout[0])
#             valve = max(0.0, valve)  # lower bound = 0
#             valve = min(100.0, valve)  # upper bound = 100
#             if valve > 100.0:  # check upper limit
#                 valve = 100.0
#                 ie[i] = ie[i] - error * delta_t  # anti-reset windup
#             if valve < 0.0:  # check lower limit
#                 valve = 0.0
#                 ie[i] = ie[i] - error * delta_t  # anti-reset windup
#
#             u1[i] = valve  # store the valve position
#             heater_board.Q1(valve)
#
#             if i % 30 == 0:
#                 df = pd.DataFrame({'time': times,
#                                    'temp': temps,
#                                    'heater_pwm': u1[0:i],
#                                    'fan_pwm': DP[0:i]})
#                 df.to_csv(box_folder_path + file_path)
#
#         # Turn off heaters
#         heater_board.Q1(0)
#         heater_board.Q2(0)
#         dpin.write(1)
#         #    heater_board.close()
#         cool = fan_cooling(dpin, heater_board)
#
#         heater_board.close()
#         board.exit()
#
#         df = pd.DataFrame({'time': times,
#                            'temp': temps,
#                            'heater_pwm': u1[0:i],
#                            'fan_pwm': DP[0:i]})
#         df.to_csv(box_folder_path + file_path)
#     #    plt.figure(z)
#     #    plt.plot(df['time'],df['temp'],'r--',linewidth=3,label='valve')
#     #    plt.ylabel('Valve')
#
#     # Allow user to end loop with Ctrl-C
#     except KeyboardInterrupt:
#         # Disconnect from Arduino
#         heater_board.Q1(0)
#         heater_board.Q2(0)
#         print('Shutting down')
#         heater_board.close()
#         dpin.write(0)
#         board.exit()
#         df = pd.DataFrame({'time': times,
#                            'temp': temps,
#                            'heater_pwm': u1[0:i],
#                            'fan_pwm': DP[0:i]})
#         df.to_csv(box_folder_path + file_path)
#
#
#     # Make sure serial connection still closes when there's an error
#     except:
#         # Disconnect from Arduino
#         heater_board.Q1(0)
#         heater_board.Q2(0)
#         print('Error: Shutting down')
#         heater_board.close()
#         dpin.write(0)
#         board.exit()
#         df = pd.DataFrame({'time': times,
#                            'temp': temps,
#                            'heater_pwm': u1[0:i],
#                            'fan_pwm': DP[0:i]})
#         df.to_csv(box_folder_path + file_path)
#         raise
