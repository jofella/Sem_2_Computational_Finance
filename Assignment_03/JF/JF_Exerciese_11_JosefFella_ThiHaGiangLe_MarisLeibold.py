# C-Exercise 10, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella, Thi Ha Giang Le

import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt



# 1.Part: BS_EuCall_MC_IS


def BS_EuCall_MC_IS(S_0, r, sigma, K, T, mu, N, alpha):
    
    # Eu call function
    def eu_call_payoff(x, K):
        return np.maximum(x - K, 0)
    
    # 1.Step: Set up data
    Y_norm_sample = np.random.normal(mu, 1, N)
    S_T = np.empty(len(Y_norm_sample), dtype = float)
    Y = np.empty(len(Y_norm_sample), dtype = float)   
    
    # 2.Step: Simulate stock price with BS (adapted by Yn dist (mu, 1))
    for i in range(0, M, 1):
        S_T[i] = S_0 * math.exp((r - 0.5 * np.power(sigma, 2)) * T + sigma * np.sqrt(T) * Y_norm_sample[i])
        adjust_factor = math.exp(-r * T - Y_norm_sample[i] * mu + 0.5 * np.power(mu, 2))
        Y[i] =  adjust_factor * eu_call_payoff(S_T[i], K)
    
    # Get V_hat_IS
    V_hat_IS = np.mean(Y)
    V_0 = math.exp(-r * T) * V_hat_IS
    
    return V_0



# Test function
S_0 = 100
r = 0.05
mu = 0.04
sigma = 0.3
T = 1
K = 220
N = 100000
alpha = 0.95


V_0 = BS_EuCall_MC_IS(S_0, r, sigma, K, T, mu, N, alpha)
print(V_0)



# 2.Part: BS_EuCall

def BlackScholes_EuCall(t, S_0, r, sigma, T, K):
    d1 = (np.log(S_0/K) + (r + (np.power(sigma, 2)/2)) * (T-t)) / (sigma * np.sqrt(T-t))
    d2 = d1 - sigma * np.sqrt(T-t)
    
    # Imported from scipy to get normal cdf value
    phi_d1 = scipy.stats.norm.cdf(d1)
    phi_d2 = scipy.stats.norm.cdf(d2)
    
    V_0  = S_0 * phi_d1 - K * math.exp(-r * (T-t)) * phi_d2    
    
    return V_0



# Test function
S_0 = 100
t = 0
r = 0.05
mu = 0.1 #supposed to be changed
sigma = 0.3
T = 1
K = 220
N = 10000
alpha = 0.95


BS_price = BlackScholes_EuCall(t, S_0, r, sigma, T, K)
V_0 = BS_EuCall_MC_IS(S_0, r, sigma, K, T, mu, N, alpha)
print(V_0)

print(BS_price)






# Plotting against changing mu




# plot 
plt.clf()
plt.plot()
plt.show()