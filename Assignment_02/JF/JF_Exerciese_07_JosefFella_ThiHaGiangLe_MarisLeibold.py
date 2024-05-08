# C-Exercise 07, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella, Thi Ha Giang Le

import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt



def Eu_Option_BS_MC (S0, r, sigma, T, K, M, f):
    return output


# BS-Call Option

def BlackScholes_EuCall(t, S_t, r, sigma, T, K):
    d1 = (np.log(S_t/K) + (r + (np.power(sigma, 2)/2)) * (T-t)) / (sigma * np.sqrt(T-t))
    d2 = d1 - sigma * np.sqrt(T-t)
    
    phi_d1 = scipy.stats.norm.cdf(d1)
    phi_d2 = scipy.stats.norm(d2)
    
    V_0  = S_t * phi_d1 - K * math.exp(-r * (T-t)) * phi_d2    
    
    return V_0