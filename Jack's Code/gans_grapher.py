import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 36}
matplotlib.rc('font', **font)
matplotlib.use('Qt5Agg')

# df = pd.read_csv("data/real.csv")
df = pd.read_csv('forecast_iter1.csv')
df1 = pd.read_csv("orig_iter6.csv")
df5 = pd.read_csv("50000.csv")

dfn = pd.read_csv("generated_iter5.csv")
dfn = dfn.dropna()
i = 10
y_real = np.array(df.iloc[i, :])
x = np.arange(0, len(y_real))

y_gans = np.array(df5.iloc[i, :])

y_opti = np.array(dfn.iloc[i, :])

plt.figure(figsize=(19, 10))
plt.plot(x, y_real, label='Forecast NREL Sample', color='red', alpha=1)
plt.plot(x, y_gans, label='GANS', color='blue', alpha=0.7)
plt.plot(x, y_opti, label='Stochastic Optimization', color='orange', alpha=0.7)
# plt.legend()
# plt.title('Progress from NREL Data to Stochastically Optimized GANS')
plt.xlabel('Time in 5 minute intervals')
plt.ylabel('Wind Power (MW)')
plt.tight_layout()
plt.savefig('gans1.png')
plt.plot()

plt.figure(figsize=(19, 10))
plt.plot(x, y_real, label='Forecast NREL Sample', color='red', alpha=1)
plt.plot(x, y_gans, label='GANS', color='blue', alpha=0.7)
plt.plot(x, y_opti, label='Stochastic Optimization', color='orange', alpha=0.7)
plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
           fancybox=True, shadow=True, ncol=1)
# plt.legend()
# plt.title('Progress from NREL Data to Stochastically Optimized GANS')
plt.xlabel('Time in 5 minute intervals')
plt.ylabel('Wind Power (MW)')
plt.tight_layout()
plt.savefig('gans1_w_legend.png')
plt.plot()

dfp = pd.read_csv(
    r'/Users/jmcmullin/Desktop/Spring 2021'
    r'/Senior Lab/Renewables_Scenario_Gen_GAN-master/pvalues_such.csv')

P_real = np.array(dfp['p_real'])
P_fake = np.array(dfp['p_fake'])
discrim_loss = np.array(dfp['discrim_loss'])
plt.figure(figsize=(19, 10))
plt.plot(P_real, label="real", color='blue')
plt.plot(P_fake, label="fake", color='red')
# plt.legend()
# plt.title('Evolution of the Discriminator Output')
plt.ylabel('D(x)/D(G(z))')
plt.xlabel('Iterations')
plt.tight_layout()
plt.savefig('gans2.png')
plt.show()

plt.figure(figsize=(19, 10))
plt.plot(P_real, label="real", color='blue')
plt.plot(P_fake, label="fake", color='red')
plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
           fancybox=True, shadow=True, ncol=1)
# plt.title('Evolution of the Discriminator Output')
plt.ylabel('D(x)/D(G(z))')
plt.xlabel('Iterations')
plt.tight_layout()
plt.savefig('gans2_w_legend.png')
plt.show()

plt.figure(figsize=(19, 10))
plt.plot(discrim_loss, label="discrim_loss", color='blue')
# plt.legend()
plt.title('Discriminator Loss over iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig('gans3.png')
plt.show()
