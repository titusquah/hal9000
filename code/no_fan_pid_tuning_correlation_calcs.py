Kp = 1.0758313073767947
taup = 312.2249176059667
thetap = 0.0
dt = 1
thetap += dt/2


tauc = max([0.1*taup,0.8*thetap])
kc = 1/Kp*taup/(thetap+tauc)
taui = taup

print('Kc: ' + str(kc))
print('taui: ' + str(taui))
