import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return 2.5*x ** 0.8


def fit_func(x, b, c):
    return c*x ** b


xs = np.linspace(0, 10)
ys1 = func(xs) + np.random.uniform(4, size=(len(xs)))

bs = np.linspace(0, 5)
cs = np.linspace(0, 5)
norms = []
for val1 in bs:
    for val2 in cs:
        ys = fit_func(xs, val1)
        diff = ys - ys1
        norm = np.linalg.norm(diff, 2)
        norms.append(norm)

plt.close('all')
plt.figure()
plt.semilogy(bs, norms)
plt.show()
plt.figure()
plt.semilogy(xs, ys1)
plt.show()
