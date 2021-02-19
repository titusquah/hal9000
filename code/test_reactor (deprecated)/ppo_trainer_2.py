from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from unstable_reactor_gym import UnstableReactor
import matplotlib.pyplot as plt
import numpy as np

# multiprocess environment
max_time = 150
dt = 1 / 60 / 5
env = UnstableReactor(dt=dt, max_time=max_time)

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)

# Enjoy trained agent
states = []
actions = []
rewards = []

obs = env.reset()
states.append(obs)
actions.append(0)
rewards.append(0)
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    states.append(obs)
    actions.append(action)
    rewards.append(reward)

fig, ax = plt.subplots(3)
time = np.arange(0, max_time + dt, dt)
ax[0].plot(time, states)
