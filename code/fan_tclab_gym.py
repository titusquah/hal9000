import numpy as np
from scipy.integrate import odeint
import pandas as pd
from collections import OrderedDict
from gym import spaces
from gym.utils import seeding
import gym


class FanTempControlLabGrayBox(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self,
                 initial_temp=296.15,
                 amb_temp=296.15,
                 cp=500,
                 surf_area=1.2e-3,
                 mass=.004,
                 emissivity=0.9,
                 dt=0.1,
                 max_time=6000,
                 alpha=0.01,
                 tau_hc=5,
                 tau_d=0,
                 k_d=1,
                 beta1=11.33,
                 beta2=0.6,
                 d_traj=None,
                 temp_lb=296.15,
                 ):

        self.initial_temp = initial_temp  # K
        self.amb_temp = amb_temp  # K
        self.cp = cp  # J / kg K
        self.surf_area = surf_area  # m^2
        self.mass = mass  # kg
        self.emissivity = emissivity  # unitless
        self.dt = dt  # s
        self.max_time = max_time  # number of time steps (10 min)
        self.alpha = alpha  # W / PWM %
        self.tau_hc = tau_hc  # s
        self.tau_d = tau_d  # s
        self.k_d = k_d  # m/s / PWM %
        self.beta1 = beta1  # nusselt pre exponential term
        self.beta2 = beta2  # nusselt power term
        self.d_traj = d_traj  # PWM %
        self.temp_lb = temp_lb
        self.boltzmann = 5.67e-8  # W/ m^2 K^4

        if self.d_traj is None:
            folder_path_txt = "hidden/box_folder_path.txt"
            with open(folder_path_txt) as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            box_folder_path = content[0]
            file_path = "/data/d_traj.csv"
            df = pd.read_csv(box_folder_path + file_path)

            start = 0
            stop = 600
            time = df['index'].values[start:stop]
            dist = np.clip(
                pd.to_numeric(df['load'], errors='coerce').values[start:stop],
                0, None)
            dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist)) * (
                    100 - 20) + 20
            self.d_traj = dist

        self.np_random = None
        self.state = None

        inf = np.finfo(np.float32).max

        # sensor temp, heater temp, fan_velocity
        high_set = [inf, inf, inf]
        low_set = [0, 0, 0]
        high = np.array(high_set)
        low = np.array(low_set)

        self.action_space = spaces.Box(low=0, high=1, shape=(1,),
                                       dtype=np.float64)

        self.observation_space = spaces.Box(low=low, high=high,
                                            dtype=np.float64)

        self.current_step = 0
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def tclab_step(self, state, time, action, dist):
        """
        Function to model TCLab with fan in odeint
        :param state: (np.ndarray)
        :param time: (np.ndarray) time to integrate function over (s)
        :param action: (float) PWM to heater (%)
        :param dist: (float) PWM to fan (%)
        :return: new_state: (np.ndarray)
        """
        heater_pwm = action
        sensor_temp, heater_temp, fan_speed = state

        dth = (self.surf_area * self.beta1 * fan_speed ** self.beta2 * (
                self.amb_temp - heater_temp) + self.emissivity * self.boltzmann
               * self.surf_area * (self.amb_temp ** 4 - heater_temp ** 4)
               + self.alpha * heater_pwm) / (self.mass * self.cp)
        duf = (-fan_speed + self.k_d * dist) / self.tau_d
        dtc = (-sensor_temp + heater_temp) / self.tau_hc

        new_state = np.zeros(3)
        new_state[0] = dth
        new_state[1] = duf
        new_state[2] = dtc
        return new_state

    def reset(self):
        self.current_step = 0
        zero = np.float64(0)
        high_set = [self.initial_temp, self.initial_temp, zero]
        low_set = [self.initial_temp, self.initial_temp, zero]
        high = np.array(high_set)
        low = np.array(low_set)

        self.state = self.np_random.uniform(low=low, high=high)
        return self.state

    def step(self, action):
        """
        :param action:(np.ndarray)
        :return:
        """
        sensor_temp, heater_temp, fan_speed = self.state
        action = np.clip(action, 0, 1)
        second = self.current_step // 10
        current_dist = self.d_traj[second]
        heater_pwm = 100 * action[0]
        inputs = tuple([heater_pwm, current_dist])
        time = [0, self.dt]
        new_sensor_temp, new_heater_temp, new_fan_speed = \
            odeint(self.tclab_step, self.state, time, inputs)[-1]

        new_sensor_temp = np.clip(new_sensor_temp, 0, None)
        new_heater_temp = np.clip(new_heater_temp, 0, None)
        new_fan_speed = np.clip(new_fan_speed, 0, None)

        self.state = [new_sensor_temp, new_heater_temp, new_fan_speed]

        if new_sensor_temp < self.temp_lb:
            penalty = -10
        else:
            penalty = 0

        reward = -heater_pwm - penalty

        self.current_step += 1

        done = self.current_step >= self.max_time  # or done
        info = {'is_success': done}
        return self.state, reward, done, info

    def render(self, mode='human'):
        print("Not implemented")
        return

    def close(self):
        print("Not implemented")
        return


class FanTempControlLabBlackBox(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self,
                 initial_temp=296.15,
                 amb_temp=296.15,
                 dt=0.1,
                 max_time=6000,
                 d_traj=None,
                 temp_lb=296.15,
                 c1=0.001,
                 c2=0.6,
                 c3=1e-2,
                 c4=0.05):
        self.initial_temp = initial_temp
        self.amb_temp = amb_temp
        self.dt = dt
        self.max_time = max_time
        self.d_traj = d_traj
        self.temp_lb = temp_lb
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4

        if self.d_traj is None:
            folder_path_txt = "hidden/box_folder_path.txt"
            with open(folder_path_txt) as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            box_folder_path = content[0]
            file_path = "/data/d_traj.csv"
            df = pd.read_csv(box_folder_path + file_path)

            start = 0
            stop = 600
            time = df['index'].values[start:stop]
            dist = np.clip(
                pd.to_numeric(df['load'], errors='coerce').values[start:stop],
                0, None)
            dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist)) * (
                    100 - 20) + 20
            self.d_traj = dist

        self.np_random = None
        self.state = None

        inf = np.finfo(np.float32).max

        # sensor temp, heater temp
        high_set = [inf, inf]
        low_set = [0, 0]
        high = np.array(high_set)
        low = np.array(low_set)

        self.action_space = spaces.Box(low=0, high=1, shape=(1,),
                                       dtype=np.float64)

        self.observation_space = spaces.Box(low=low, high=high,
                                            dtype=np.float64)

        self.current_step = 0
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def tclab_step(self, state, time, action, dist):
        """
        Function to model TCLab with fan in odeint
        :param state: (np.ndarray)
        :param time: (np.ndarray) time to integrate function over (s)
        :param action: (float) PWM to heater (%)
        :param dist: (float) PWM to fan (%)
        :return: new_state: (np.ndarray)
        """
        heater_pwm = action
        sensor_temp, heater_temp = state
        c1 = self.c1
        c2 = self.c2
        c3 = self.c3
        c4 = self.c4
        amb_temp = self.amb_temp

        dth = -c1 * dist ** (
                c2 - 1) * heater_temp + c3 * heater_pwm + c1 * c2 * dist ** (
                c2 - 1) * (amb_temp - heater_temp) * dist
        dtc = c4*heater_temp-c4*sensor_temp

        new_state = np.zeros(2)
        new_state[0] = dth
        new_state[1] = dtc
        return new_state

    def reset(self):
        self.current_step = 0
        zero = np.float64(0)
        high_set = [self.initial_temp, self.initial_temp, zero]
        low_set = [self.initial_temp, self.initial_temp, zero]
        high = np.array(high_set)
        low = np.array(low_set)

        self.state = self.np_random.uniform(low=low, high=high)
        return self.state

    def step(self, action):
        """
        :param action:(np.ndarray)
        :return:
        """
        sensor_temp, heater_temp = self.state
        action = np.clip(action, 0, 1)
        second = self.current_step // 10
        current_dist = self.d_traj[second]
        heater_pwm = 100 * action[0]
        inputs = tuple([heater_pwm, current_dist])
        time = [0, self.dt]
        new_sensor_temp, new_heater_temp = \
            odeint(self.tclab_step, self.state, time, inputs)[-1]

        new_sensor_temp = np.clip(new_sensor_temp, 0, None)
        new_heater_temp = np.clip(new_heater_temp, 0, None)

        self.state = [new_sensor_temp, new_heater_temp]

        if new_sensor_temp < self.temp_lb:
            penalty = -10
        else:
            penalty = 0

        reward = -heater_pwm - penalty

        self.current_step += 1

        done = self.current_step >= self.max_time  # or done
        info = {'is_success': done}
        return self.state, reward, done, info

    def render(self, mode='human'):
        print("Not implemented")
        return

    def close(self):
        print("Not implemented")
        return
