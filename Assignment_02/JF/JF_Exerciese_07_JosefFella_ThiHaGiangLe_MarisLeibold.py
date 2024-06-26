# C-Exercise 07, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella, Thi Ha Giang Le


### Corrected version of 07

import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# Fix random seed
np.random.seed(28) # my favorite number



# Option function
def f(x, K):
    return np.maximum(x - K, 0) # more dynamic with K input function

# MC function
def Eu_Option_BS_MC(S_0, r, sigma, T, M, f):
   
   # 1.Step: Set up data
    # Generate normal sample and container
   X = np.random.normal(0, 1, M)
   S_T = np.empty(len(X), dtype = float)
   Y = np.empty(len(X), dtype = float) # undiscounted Option prices
   
   # 2.Step: Simulate stock price with BS
   for i in range(0, M, 1):
        S_T[i] = S_0 * math.exp((r - 0.5 * np.power(sigma, 2)) * T + sigma * np.sqrt(T) * X[i]) # forgot 0.5
        Y[i] = f(S_T[i], K) #going thru all stock prices, otherwise constant!!!
        
   # 3.Step: Calculate V_hat (estimator) and V_0
   V_hat = np.mean(Y)
   V_0 = math.exp(-r * T) * V_hat
   
   # 4.Step: Calculate 95% CI (skript p.13) -- corrected due to mistake
   
    # Sample variance
   epsilon = 1.96 * math.sqrt(np.var(Y) / M) #from solutions
   
   c1 = math.exp(-r * T) * (V_hat - epsilon)
   c2 = math.exp(-r * T) * (V_hat + epsilon)

   return V_0, V_hat, c1, c2


# Test function
S_0 = 110
r = 0.04
sigma = 0.2
T = 1
M = 10000
K = 100


V_0, V_hat, c1, c2 = Eu_Option_BS_MC(S_0, r, sigma, T, M, f)
print(V_0, V_hat, c1, c2)


# 2.Step: Get  BS call price -- From 01

def BlackScholes_EuCall(t, S_0, r, sigma, T, K):
    d1 = (np.log(S_0/K) + (r + (np.power(sigma, 2)/2)) * (T-t)) / (sigma * np.sqrt(T-t))
    d2 = d1 - sigma * np.sqrt(T-t)
    
    # Imported from scipy to get normal cdf value
    phi_d1 = scipy.stats.norm.cdf(d1)
    phi_d2 = scipy.stats.norm.cdf(d2)
    
    V_0  = S_0 * phi_d1 - K * math.exp(-r * (T-t)) * phi_d2    
    
    return V_0


# 3.Step: Compare prices

# Set parameter
t = 0
S_0 = 110
r = 0.04
sigma = 0.2
T = 1
K = 100
M = 10000
no_MC_call_prices = 100 # carefully chose this

# Set up container
Eu_Option_BS_MC_price = Eu_Option_BS_MC(S_0, r, sigma, T, M, f)
BlackScholes_EuCall_price = BlackScholes_EuCall(t, S_0, r, sigma, T, K)

# Get 100 MC call prices
V_0__MC_EuCall = np.empty(no_MC_call_prices, dtype=float)
V_0__BS_EuCall = np.empty(no_MC_call_prices, dtype=float)

# Idea is to simulate many MC prices and compare it to the BS model
for i in range(0, no_MC_call_prices, 1):
    V_0__MC_EuCall[i], _, _ = Eu_Option_BS_MC(S_0, r, sigma, T, M, f) # Removing the CI
    V_0__BS_EuCall[i] = BlackScholes_EuCall(t, S_0, r, sigma, T, K)


# Output terminal message
print(f"The MC price is: {Eu_Option_BS_MC_price[0]} with CI form {Eu_Option_BS_MC_price[1]} to \
{Eu_Option_BS_MC_price[2]} and the \
BS price is: {BlackScholes_EuCall_price}. \
The difference is {BlackScholes_EuCall_price - Eu_Option_BS_MC_price[0]}")


# Plot Error for 1000 different simulations
plt.clf()
plt.plot(V_0__BS_EuCall - V_0__MC_EuCall, 'r')

plt.title('Difference of BS and MC')
plt.xlabel('simulation')
plt.ylabel('error')

plt.legend()
plt.show()