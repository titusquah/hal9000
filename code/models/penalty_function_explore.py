import numpy as np
import matplotlib.pyplot as plt


def penalty(t, k):
    # return k * t / (k - t + 1)



t = np.linspace(-1, 1)
k = 0.2

p = penalty(t, k)


plt.close('all')
plt.figure()
plt.plot(t, p)
plt.show()
