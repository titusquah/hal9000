# Python code for 1-D random walk.
import random
import numpy as np
import matplotlib.pyplot as plt

# Probability to move up or down
prob = [0.2, 0.8]
np.random.seed(10)
# statically defining the starting position
for i in range(100):
    start = 20
    positions = [start]

    # creating the random points
    rr = np.random.default_rng(123).random(100)
    downp = rr < prob[0]
    upp = rr > prob[1]

    for idownp, iupp in zip(downp, upp):
        down = idownp and positions[-1] > 0
        up = iupp and positions[-1] < 100
        positions.append(positions[-1] - down + up)

    n_pos = positions / max(positions)
    # print(n_pos)
    # plotting down the graph of the random walk in 1D
    plt.plot(n_pos, alpha=0.5)
plt.grid()
plt.show()
