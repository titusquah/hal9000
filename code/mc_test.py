import numpy as np
import matplotlib.pyplot as plt

min_val = 0
max_val = 1
init_val = 0.2 + 0.0001

fig, ax = plt.subplots(2)
for i in range(1000):
    steps = np.arange(0, 100)
    sigma = (max_val - min_val) * 0.001
    sigmas = [sigma]
    vals = [init_val]
    for step in steps:
        if step == 0:
            continue
        # sigma = np.clip((1 - vals[-1]) * vals[-1], 0, 1)
        sigma = np.abs(
            (0.4 - vals[-1]) * (0.6 - vals[-1]) * (0.8 - vals[-1]))*2
        kick = False
        if sigma < 1e-4:
            kick = True
        new_val = np.clip(np.random.normal(vals[-1] - kick * 0.1, sigma), 0, 1)
        vals.append(new_val)
        sigmas.append(sigma)
    ax[0].plot(steps, vals, alpha=0.5)
    ax[1].plot(steps, sigmas, alpha=0.5)
plt.show()
