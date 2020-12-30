from unstable_reactor_gym import UnstableReactor
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 0.5, 401)
dt = t[1] - t[0]
reactor = UnstableReactor(dt=dt)
actions = [280]
states = []
state = reactor.reset()
states.append(state)

for i in range(len(t) - 1):
    state, reward, done, info = reactor.step(1)
    states.append(state)
    actions.append(350)
states = np.array(states)

plt.figure(figsize=(10, 7))
plt.subplot(3, 1, 1)
plt.plot(t, actions, 'b--', linewidth=3)

plt.ylabel('Cooling T (K)')
plt.legend(['Jacket Temperature'], loc='best')

plt.subplot(3, 1, 2)
plt.plot(t, states[:, 0], 'b.-', linewidth=3, label=r'$C_A$')
plt.plot([0, t[-1]], [0.2, 0.2], 'r--', linewidth=2, label='limit')
plt.ylabel(r'$C_A$ (mol/L)')
plt.legend(loc='best')

plt.subplot(3, 1, 3)
plt.plot(t, states[:, 2], 'b.-', linewidth=3, label=r'$T_{meas}$')
plt.plot([0, t[-1]], [400, 400], 'r--', linewidth=2, label='limit')
plt.ylabel('T (K)')
plt.xlabel('Time (min)')
plt.legend(loc='best')
plt.show()
