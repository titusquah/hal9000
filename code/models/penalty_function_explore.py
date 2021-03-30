import numpy as np
import matplotlib.pyplot as plt


def penalty(t, k):
    return np.log(1+np.exp(k*t))/k
    # return k * t / (k - t + 1)

temp_lb = 310

temp = np.linspace(temp_lb-1, temp_lb+2, 1000)
k = 10
diff = temp_lb-temp
p = penalty(diff, k)


plt.close('all')
plt.figure()
plt.plot(diff, 1445*p)
plt.plot(diff, np.zeros(len(diff)))
plt.show()

# counter = np.arange(0, 24)
# max_change = 0.8
# min_change = 0.02
# rate = 0.25
# values = max_change*np.exp(-rate*counter)+min_change
# plt.figure()
# plt.plot(counter,values)

