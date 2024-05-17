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
    for i in range(0, N, 1):
        S_T[i] = S_0 * math.exp((r - 0.5 * np.power(sigma, 2)) * T + sigma * np.sqrt(T) * Y_norm_sample[i])
        adjust_factor = math.exp(-r * T - Y_norm_sample[i] * mu + 0.5 * np.power(mu, 2))
        Y[i] =  adjust_factor * eu_call_payoff(S_T[i], K) # ref. 2.9
    
    # Get V_hat_IS
    V_hat_IS = np.mean(Y)
    V_0 = math.exp(-r * T) * V_hat_IS
    
    # 3. Step: Calculate the CI
    z_score = scipy.stats.norm.ppf((1 - alpha) / 2) # from scipy doc, get score value for alpha
    Var_hat_IS = 1/(N-1) *  (np.mean(np.power(V_hat_IS, 2)) - np.power(V_hat_IS, 2))
    
    epsilon = 1.96 * math.sqrt(Var_hat_IS / N) # ref. solutions
    
    # Get CI
    CIl = math.exp(-r * T) * (Var_hat_IS - epsilon) #somehow wrong here
    CIl = math.exp(-r * T) * (Var_hat_IS + epsilon)

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



# 3.Part: Visiualize data


# Test function
S_0 = 100
t = 0
r = 0.05
mu = np.arange(-4, 4, 0.1)
sigma = 0.3
T = 1
K = 220
N = 10000
alpha = 0.95


# Set up data
BS_MC_IS_prices = np.empty(len(mu))

for i in range(len(mu)):
    BS_MC_IS_prices[i] = BS_EuCall_MC_IS(S_0, r, sigma, K, T, mu[i], N, alpha)
BS_price = BlackScholes_EuCall(t, S_0, r, sigma, T, K)
print(BS_MC_IS_prices)



# plot 
plt.clf()
plt.plot(mu, BS_MC_IS_prices, color='b',label='BS_MC_IS_prices')
plt.axhline(xmin=0, y = BS_price, color = 'r', linestyle = '-', label='BS EuPut')

plt.title('BS_MC_IS_prices versus true BS value conversion')
plt.xlabel('$\mu$ value')
plt.ylabel('option price')

plt.legend()
plt.show()