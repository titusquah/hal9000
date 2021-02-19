Kp = 1.0758313073767947
taup = 312.2249176059667
thetap = 0.0
dt = 1
thetap += dt/2


tauc = max([0.1*taup,0.8*thetap])
kc = 1/Kp*taup/(thetap+tauc)
taui = taup
print("IMC Aggressive constants")
print('Kc: ' + str(kc))
print('taui: ' + str(taui))
print()
kc = 0.586/Kp*(thetap/taup)**(-0.916)
taui = taup/(1.03-0.165*(thetap/taup))
print("ITAE Tuning correlations")
print('Kc: ' + str(kc))
print('taui: ' + str(taui))
