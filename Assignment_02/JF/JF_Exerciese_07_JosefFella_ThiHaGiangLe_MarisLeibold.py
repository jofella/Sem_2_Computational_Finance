# C-Exercise 07, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella, Thi Ha Giang Le

import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt



def Eu_Option_BS_MC (S_0, r, sigma, T, K, M, f):
    
    # Generates normal sample
    X = np.random.normal(0, 1, M)
     
    S_T = S_0 * math.exp((r - np.power(sigma, 2)) * T + sigma * np.sqrt(T) * X)
    V_0 = math.exp(-r*T) * np.mean(some_dings_bums)
    return 




# BS-Call Option

def BlackScholes_EuCall(t, S_t, r, sigma, T, K):
    d1 = (np.log(S_t/K) + (r + (np.power(sigma, 2)/2)) * (T-t)) / (sigma * np.sqrt(T-t))
    d2 = d1 - sigma * np.sqrt(T-t)
    
    # Imported from scipy to get normal cdf value
    phi_d1 = scipy.stats.norm.cdf(d1)
    phi_d2 = scipy.stats.norm.cdf(d2)
    
    V_0  = S_t * phi_d1 - K * math.exp(-r * (T-t)) * phi_d2    
    
    return V_0



# Test function
t = 1
S_0 = 100
r = 0.04
sigma = 0.2
T = 1
K = 130 # the closer ITM (in the money) the higher the price


V_0_BS = BlackScholes_EuCall(t, S_0, r, sigma, T, K)
print(V_0_BS)