import pandas as pd
import matplotlib.pyplot as plt

test = 'perfect'
case = 1


file_path = "/data/real_{0}_test_case_{1}(4).csv".format(test,case)
file_path = "/data/real_perfect_test_step(8).csv"
folder_path_txt = "../hidden/box_folder_path.txt"
with open(folder_path_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]
box_folder_path = content[0]

df = pd.read_csv(box_folder_path + file_path)
tlb = df['temp_lb'][0]

fig, ax = plt.subplots(4)
ax[0].plot(df.time,
           df.temp, 'bo', label='Measured', markersize=2)
ax[0].plot(df.time,
           df.est_temp, 'ro', label='Predicted', markersize=2)
ax[0].axhline(tlb, color='b', label='$T_{lb}$')
ax[0].legend()
ax[1].plot(df.time,
           df.fan_pwm, 'b-', label='Fan PWM')
ax[2].plot(df.time,
           df.heater_pwm, 'r-', label='Heater PWM')
ax[3].plot(df.time, df.c1, 'g-', label='$c_1$')
ax[3].plot(df.time, df.c2, 'r-', label='$c_2$')
ax[3].plot(df.time, df.c3, 'b-', label='$c_3$')
ax[3].plot(df.time, df.c4, 'k-', label='$c_4$')
ax[3].legend()
plt.show()