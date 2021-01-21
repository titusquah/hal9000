from unstable_reactor_gym import UnstableReactor
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 0.5, 0.004)
dt = t[1] - t[0]
reactor = UnstableReactor(dt=dt,
                          initial_tank_conc_min=1.,
                          initial_tank_conc_max=1.,
                          initial_tank_temp_min=304,
                          initial_tank_temp_max=304,
                          max_time=10,
                          hold_time=10,
                          rel_diff_for_main_reset=1e-2
                          )

actions = [280]
states = []
state = reactor.reset()
states.append(state)
finished = False
while not finished:
    done = False
    ind1 = 0
    while not done:
        state, reward, done, info = reactor.step([1])
        if ind1 == 0:
            states.append(state)
            actions.append(350)
        ind1 += 1
    reactor.reset()
    finished = reactor.main_reset


states = np.array(states)
t = [ind*dt for ind in range(len(actions))]
fig, ax = plt.subplots(3, figsize=(10, 7))
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
