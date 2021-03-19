import pandas as pd
import matplotlib.pyplot as plt

# for i in range(1, 7):
case = 3

file_path = "/data/feedforward_{}.csv".format(case)
# file_path = "/data/real_perfect_test_step(9).csv"
folder_path_txt = "../hidden/box_folder_path.txt"
with open(folder_path_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]
box_folder_path = content[0]

df = pd.read_csv(box_folder_path + file_path)
tlb = 30

fig, ax = plt.subplots(3)
ax[0].plot(df.time,
           df.temp, 'bo', label='Measured', markersize=2)
ax[0].axhline(tlb, color='b', label='$T_{lb}$')
ax[0].legend()
ax[1].plot(df.time,
           df.fan_pwm, 'b-', label='Fan PWM')
ax[2].plot(df.time,
           df.heater_pwm, 'r-', label='Heater PWM')
plt.show()
