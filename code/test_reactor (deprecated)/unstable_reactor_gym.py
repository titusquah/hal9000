import numpy as np
from scipy.integrate import odeint

from collections import OrderedDict
from gym import spaces
from gym.utils import seeding
import gym


class UnstableReactor(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self,
                 initial_jacket_temp_min=280,
                 initial_jacket_temp_max=280,
                 feed_temp=350,
                 feed_conc=1,
                 feed_flowrate=100,
                 tank_vol=100,
                 rho=1000,
                 cp=0.239,
                 dh=5e4,
                 e_over_r=8750,
                 preexp_factor=7.2e10,
                 ua=5e4,
                 initial_tank_conc_min=1,
                 initial_tank_conc_max=1,
                 initial_tank_temp_min=304,
                 initial_tank_temp_max=304,
                 dt=1,
                 max_jacket_temp=350,
                 min_jacket_temp=250,
                 n_look_back=0,
                 max_time=500,
                 hold_time=5,
                 tank_temp_ub=400,
                 tank_conc_ub=0.2,
                 rel_diff_for_main_reset=1e-4,
                 ):
        self.initial_jacket_temp_min = initial_jacket_temp_min  # K
        self.initial_jacket_temp_max = initial_jacket_temp_max  # K
        self.feed_temp = feed_temp  # K
        self.feed_conc = feed_conc  # mol/m^3
        self.feed_flowrate = feed_flowrate  # m^3/sec
        self.tank_vol = tank_vol  # m^3
        self.rho = rho  # kg/m^3
        self.cp = cp  # J/mol
        self.dh = dh  # J/mol
        self.e_over_r = e_over_r  # K
        self.preexp_factor = preexp_factor  # 1/sec
        self.ua = ua  # W/K
        self.initial_tank_conc_min = initial_tank_conc_min  # mol/m^3
        self.initial_tank_temp_min = initial_tank_temp_min  # K
        self.initial_tank_conc_max = initial_tank_conc_max  # mol/m^3
        self.initial_tank_temp_max = initial_tank_temp_max  # K

        self.dt = dt
        self.jacket_temp_max = max_jacket_temp  # K
        self.jacket_temp_min = min_jacket_temp  # K
        self.jacket_temp_range = max_jacket_temp - min_jacket_temp  # K
        self.n_look_back = n_look_back
        self.max_time = max_time
        self.hold_time = hold_time
        self.tank_temp_ub = tank_temp_ub
        self.tank_conc_ub = tank_conc_ub
        self.rel_diff_for_main_reset = rel_diff_for_main_reset

        self.np_random = None
        self.state = None
        self.main_reset = True
        self.last_tank_conc = None
        self.last_tank_temp = None
        self.tank_conc_stable_counter = 0
        self.tank_temp_stable_counter = 0

        inf = np.finfo(np.float32).max

        # C_a, Temp, mini_time
        high_set = [inf, 6e6, self.max_time]
        low_set = [0, 0, 0]
        high = []
        low = []
        for i in range(self.n_look_back + 1):
            high.extend(high_set)
            low.extend(low_set)
        high = np.array(high)
        low = np.array(low)

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

    def cstr_step(self, state, time, action, feed_temp, feed_conc):
        """
        Function to model CSTR in odeint
        :param state: (np.ndarray)
        :param time: (np.ndarray) time to integrate function over
        :param action: (float) Temperature of cooling jacket (K)
        :param feed_temp: (float) Feed Temperature (K)
        :param feed_conc: (float) Feed Concentration (mol/m^3)
        :return: new_state: (np.ndarray)
        """
        jacket_temp = action
        tank_conc_a = state[0]
        # tank_conc_b = state[1]
        tank_temp = state[1]
        feed_flowrate = self.feed_flowrate
        tank_vol = self.tank_vol  # m^3
        rho = self.rho  # kg/m^3
        cp = self.cp  # J/mol
        dh = self.dh  # J/mol
        e_over_r = self.e_over_r  # K
        preexp_factor = self.preexp_factor  # 1/sec
        ua = self.ua  # W/K
        react_rate = preexp_factor * np.exp(
            -e_over_r / tank_temp) * tank_conc_a

        dconc_a_dt = feed_flowrate / tank_vol * (
                feed_conc - tank_conc_a) - react_rate
        # dconc_b_dt = feed_flowrate / tank_vol * (
        #         0 - tank_conc_b) + react_rate
        dtemp_dt = feed_flowrate / tank_vol * (feed_temp - tank_temp) \
                   + dh / (rho * cp) * react_rate \
                   + ua / tank_vol / rho / cp * (jacket_temp - tank_temp)

        new_state = np.zeros(2)
        new_state[0] = dconc_a_dt
        # new_state[1] = dconc_b_dt
        new_state[1] = dtemp_dt
        return new_state

    def reset(self):
        self.current_step = 0
        zero = np.float64(0)
        if self.main_reset:
            high_set = [self.initial_tank_conc_max,
                        self.initial_tank_temp_max, zero]
            low_set = [self.initial_tank_conc_min,
                       self.initial_tank_temp_min, zero]
        else:
            high_set = [self.last_tank_conc,
                        self.last_tank_temp, zero]
            low_set = [self.last_tank_conc,
                       self.last_tank_temp, zero]
        high = []
        low = []
        for i in range(self.n_look_back + 1):
            high.extend(high_set)
            low.extend(low_set)
        high = np.array(high)
        low = np.array(low)

        self.state = self.np_random.uniform(low=low, high=high)
        return self.state

    def step(self, action):
        """
        :param action:(np.ndarray)
        :return:
        """
        tank_conc_a, tank_temp, mini_time = self.state[0:3]
        action = np.clip(action, 0, 1)
        jacket_temp = self.jacket_temp_min + self.jacket_temp_range * action[0]
        inputs = tuple([jacket_temp, self.feed_temp, self.feed_conc])
        time = [0, self.dt]
        new_CA, new_T = \
            odeint(self.cstr_step, self.state[0:2], time, inputs)[-1]
        if new_CA < 0:
            new_CA = 0
        if new_T < 0:
            new_T = 0
        new_mini_time = self.current_step + 1

        stacks = self.state[0:-3]
        self.state = [new_CA, new_T, new_mini_time]
        self.state.extend(stacks)
        self.state = np.array(self.state)

        cooling_reward = -action
        if mini_time == 1:
            self.last_tank_conc = tank_conc_a
            self.last_tank_temp = tank_temp
            conc_diff = np.abs(tank_conc_a-new_CA)
            temp_diff = np.abs(tank_temp - new_T)
            conc_tol = new_CA*self.rel_diff_for_main_reset
            temp_tol = new_T * self.rel_diff_for_main_reset
            if new_CA < self.tank_conc_ub:
                if conc_diff < conc_tol:
                    self.tank_conc_stable_counter += 1
                else:
                    self.tank_conc_stable_counter = 0
                if temp_diff < temp_tol:
                    self.tank_temp_stable_counter += 1
                else:
                    self.tank_temp_stable_counter = 0
            if (self.tank_temp_stable_counter >= self.hold_time
                    or self.tank_conc_stable_counter >= self.hold_time):
                self.main_reset = True
            else:
                self.main_reset = False

        if new_CA < self.tank_conc_ub:
            conc_a_reward = 0
        else:
            conc_a_reward = self.tank_conc_ub - new_CA
        if new_T < self.tank_temp_ub:
            temp_reward = 0
        else:
            temp_reward = self.tank_temp_ub - new_T

        reward = 0 * cooling_reward + 10 * conc_a_reward + temp_reward

        self.current_step += 1

        done = self.current_step >= self.max_time  # or done
        #      done=False
        info = {'is_success': done,
                'jacket_temp': jacket_temp}
        return self.state, reward, done, info

    def render(self, mode='human'):
        print("Not implemented")
        return

    def close(self):
        print("Not implemented")
        return
