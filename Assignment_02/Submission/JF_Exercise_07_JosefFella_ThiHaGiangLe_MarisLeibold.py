""""----------------------------------------------------------------
Group Information
----------------------------------------------------------------"""

# Group name: QF Group 02

# Member:
# Josef Fella
# Thi Ha Giang Le
# Maris Leibold

import numpy as np
import math
import scipy.stats
import matplotlib.pyplot as plt

np.random.seed(28)
def EU_Option_BS_MC(S0, r, sigma, T, K, M, f):
    #Generate BS EU Call Option price for t=0
    d_1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) *T) / (sigma * T**0.5)
    d_2 = d_1 - sigma * T**0.5
    phi = scipy.stats.norm.cdf(d_1)
    V0_BS = S0 * phi - K * math.exp(-r * T) * scipy.stats.norm.cdf(d_2)
    #Setup for MC
    X = np.random.normal(0, 1, size=M)
    S_T = S0*np.exp(((r - sigma**2)/2)*T+sigma*T**0.5*X)
    #Calculate payoff with provided function
    V0_MC = math.exp(-r*T)*np.mean(f(S_T, K))
    #95% confidence interval
    s_var = np.var(S_T)
    CI95 = [V0_MC - 1.96 * math.sqrt(s_var/M), V0_MC + 1.96 * math.sqrt(s_var/M)]
    return V0_BS, V0_MC, CI95

def call_payoff(S_T, K):
    return np.maximum(0, S_T-K)


result = EU_Option_BS_MC(110, 0.04, 0.2, 1, 100, 10000, call_payoff)


print("BS price: " + str(result[0]))
print("MC price: " + str(result[1]))
print("MC confidence interval: " + str(result[2]))