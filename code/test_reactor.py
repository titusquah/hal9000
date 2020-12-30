import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Steady State Initial Condition
u_ss = 280.0
# Feed Temperature (K)
Tf = 350
# Feed Concentration (mol/m^3)
Caf = 1

# Steady State Initial Conditions for the States
Ca_ss = 1
T_ss = 304
x0 = np.empty(2)
x0[0] = Ca_ss
x0[1] = T_ss

# Volumetric Flowrate (m^3/sec)
q = 100
# Volume of CSTR (m^3)
V = 100
# Density of A-B Mixture (kg/m^3)
rho = 1000
# Heat capacity of A-B Mixture (J/kg-K)
Cp = 0.239
# Heat of reaction for A->B (J/mol)
mdelH = 5e4
# E - Activation energy in the Arrhenius Equation (J/mol)
# R - Universal Gas Constant = 8.31451 J/mol-K
EoverR = 8750
# Pre-exponential factor (1/sec)
k0 = 7.2e10
# U - Overall Heat Transfer Coefficient (W/m^2-K)
# A - Area - this value is specific for the U calculation (m^2)
UA = 5e4

# initial conditions
Tc0 = 280


# %% define CSTR model
def cstr(x, t, u, Tf, Caf):
    # Inputs (3):
    # Temperature of cooling jacket (K)
    Tc = u
    # Tf = Feed Temperature (K)
    # Caf = Feed Concentration (mol/m^3)

    # States (2):
    # Concentration of A in CSTR (mol/m^3)
    Ca = x[0]
    # Temperature in CSTR (K)
    T = x[1]

    # Parameters:
    # Volumetric Flowrate (m^3/sec)
    q = 100
    # Volume of CSTR (m^3)
    V = 100
    # Density of A-B Mixture (kg/m^3)
    rho = 1000
    # Heat capacity of A-B Mixture (J/kg-K)
    Cp = 0.239
    # Heat of reaction for A->B (J/mol)
    mdelH = 5e4
    # E - Activation energy in the Arrhenius Equation (J/mol)
    # R - Universal Gas Constant = 8.31451 J/mol-K
    EoverR = 8750
    # Pre-exponential factor (1/sec)
    k0 = 7.2e10
    # U - Overall Heat Transfer Coefficient (W/m^2-K)
    # A - Area - this value is specific for the U calculation (m^2)
    UA = 5e4
    # reaction rate
    rA = k0 * np.exp(-EoverR / T) * Ca

    # Calculate concentration derivative
    dCadt = q / V * (Caf - Ca) - rA
    # Calculate temperature derivative
    dTdt = q / V * (Tf - T) \
           + mdelH / (rho * Cp) * rA \
           + UA / V / rho / Cp * (Tc - T)

    # Return xdot:
    xdot = np.zeros(2)
    xdot[0] = dCadt
    xdot[1] = dTdt
    return xdot


# Time Interval (min)
t = np.linspace(0, 0.5, 401)

# Store results for plotting
Ca = np.ones(len(t)) * Ca_ss
T = np.ones(len(t)) * T_ss
Tsp = np.ones(len(t)) * T_ss
u = np.ones(len(t)) * u_ss

# Set point steps
Tsp[0:100] = 330.0
Tsp[100:200] = 350.0
Tsp[200:300] = 370.0
Tsp[300:] = 390.0

# Create plot

# plt.ion()
# plt.show()

# Simulate CSTR
for i in range(len(t) - 1):
    # simulate one time period (0.05 sec each loop)
    ts = [t[i], t[i + 1]]
    y = odeint(cstr, x0, ts, args=(u[i], Tf, Caf))
    # retrieve measurements
    Ca[i + 1] = y[-1][0]
    T[i + 1] = y[-1][1]

    u[i + 1] = 350
    # update initial conditions
    x0[0] = Ca[i + 1]
    x0[1] = T[i + 1]

    # %% Plot the results
plt.figure(figsize=(10, 7))
# plt.clf()
plt.subplot(3, 1, 1)
plt.plot(t[0:i], u[0:i], 'b--', linewidth=3)

plt.ylabel('Cooling T (K)')
plt.legend(['Jacket Temperature'], loc='best')

plt.subplot(3, 1, 2)
plt.plot(t[0:i], Ca[0:i], 'b.-', linewidth=3, label=r'$C_A$')
plt.plot([0, t[i - 1]], [0.2, 0.2], 'r--', linewidth=2, label='limit')
plt.ylabel(r'$C_A$ (mol/L)')
plt.legend(loc='best')

plt.subplot(3, 1, 3)
plt.plot(t[0:i], Tsp[0:i], 'k-', linewidth=3, label=r'$T_{sp}$')
plt.plot(t[0:i], T[0:i], 'b.-', linewidth=3, label=r'$T_{meas}$')
plt.plot([0, t[i - 1]], [400, 400], 'r--', linewidth=2, label='limit')
plt.ylabel('T (K)')
plt.xlabel('Time (min)')
plt.legend(loc='best')
# plt.draw()
# plt.pause(0.01)
plt.show()


