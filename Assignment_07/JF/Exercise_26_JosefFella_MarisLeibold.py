# C-Exercise 26, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella

import math
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt




# Function
def Heston_PCall_Lapl(S0, r, gam0, kappa, lamb, sig_tild, T, K, R, P):
    
    
    return










#b) visuals


#define payout function
def payoff(x):
    return np.maximum(x-110, 0)

Deltas = np.empty(0)
for S0 in range(60, 141):
    Deltas = np.append(Deltas, BS_Greeks_num(0.05, 0.3, S0, 1, payoff, 0.001)[0])
plt.clf()
plt.plot(range(60, 141),Deltas)
plt.title("Delta for call option")
plt.xlabel("Initial Stockprice")
plt.ylabel("Delta")
plt.show()


