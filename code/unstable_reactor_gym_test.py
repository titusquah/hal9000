from unstable_reactor_gym import UnstableReactor
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 0.5, 0.004)
dt = t[1] - t[0]
reactor = UnstableReactor(dt=dt,
                          initial_tank_conc_min=0,
                          initial_tank_conc_max=1,
                          initial_tank_temp_min=280,
                          initial_tank_temp_max=400,
                          )
fig, ax = plt.subplots(3, figsize=(10, 7))
for ind1 in range(100):
    actions = [280]
    states = []
    state = reactor.reset()
    states.append(state)

    for i in range(len(t) - 1):
        state, reward, done, info = reactor.step([1])
        states.append(state)
        actions.append(350)
    states = np.array(states)

    ax[0].plot(t, actions, 'b--', linewidth=3, alpha=0.5)

    ax[0].set_ylabel('Cooling T (K)')
    ax[0].legend(['Jacket Temperature'], loc='best')

    ax[1].plot(t, states[:, 0], 'b-', linewidth=3, label=r'$C_A$', alpha=0.5)
    ax[1].plot([0, t[-1]], [0.2, 0.2], 'r--', linewidth=2, label='limit')
    ax[1].set_ylabel(r'$C_A$ (mol/L)')
    ax[1].legend(loc='best')

    ax[2].plot(t, states[:, 1], 'b-', linewidth=3, label=r'$T_{meas}$',
               alpha=0.5)
    ax[2].plot([0, t[-1]], [400, 400], 'r--', linewidth=2, label='limit')
    ax[2].set_ylabel('T (K)')
    ax[2].set_xlabel('Time (min)')
    ax[2].legend(loc='best')
plt.show()
