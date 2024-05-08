# C-Exercise 06, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella, Thi Ha Giang Le


import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


# a) -- Our corrected version of Ass.01

# 1.Step: Get CRR stock prices
def CRR_stock(S_0, r, sigma, T, M):
    # Get parameters
    delta_t = T / M
    beta = 0.5 * (math.exp(-r * delta_t) + math.exp(((r + np.power(sigma, 2)) * delta_t)))
    u = beta + np.sqrt(np.power(beta, 2) - 1)
    d = 1 / u
    
    # Set up Matrix
    S_ji = np.empty((M + 1, M + 1)) # a bit "faster" than np.zeros
    
    # Compute stock prices
    for i in range(M + 1):
        for j in range(i + 1):
            S_ji[j, i] = S_0 * np.power(u, j) * np.power(d, i - j)
    
    return (S_ji, u, d) # more convinient for 2.Step


# 2.Step: Compute Put Option:
def CRR_AmEuPut(S_0, r, sigma, T, M, K, EU):
    # Set u, d, q -- Ref. our CRR-stock-function
    delta_t = T / M
    S_ji, u, d = CRR_stock(S_0, r, sigma, T, M)
    q = (math.exp(r * delta_t) - d) / (u - d)
     
    # Set up call price matrix
    V_ji = np.empty((M + 1, M + 1))
    
    # Get put prices at T (last column of matrix)
    V_ji[:, M] = np.maximum(K - S_ji[:, M], 0)  #### Attention here (Must be changed later)
    
    # Clean up code
    discount_factor = math.exp(-r * delta_t)
    
    ## Condition check
    # EuPut: Following skript 1.14
    if EU == 1:
        for i in range(M, 0, -1):
            for j in range(i):
                V_ji[j, i-1] = discount_factor * (q * V_ji[j+1, i]+ (1-q) * V_ji[j, i]) # watch out fliped values!!!
        return V_ji[0,0]

    # AmPut: Following skript 1.15/16
    elif EU == 0:
        
        for i in range(M, 0, -1):
            for j in range(i):
                AM_put_value = np.maximum(K - S_ji[j, i-1], 0)
                
                # Snell envelope
                V_ji[j, i-1] = np.maximum(AM_put_value, discount_factor * (q * V_ji[j+1, i]+ (1-q) * V_ji[j, i]))
        return V_ji[0,0]
            
    # Madatory elsewise we get into infinite loop    
    else:
        print('Your input must be 0 or 1')


# Test function
S_0 = 1
r = 0.05
sigma = 0.3**0.5
T = 3
M = 3
K = 1.2
EU = 0

# Print results
Test_American = CRR_AmEuPut(S_0, r, sigma, T, M, K, EU)
print(Test_American)


# b) -- Essentially similar to Ass.01

def BlackScholes_EuPut(t, S_0, r, sigma, T, K):
    d1 = (np.log(S_0 / K) + (r + (np.power(sigma, 2) / 2)) * (T-t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)
    
    # Calculate phi's
    phi_d1 = scipy.stats.norm.cdf(-d1) # minus for put
    phi_d2 = scipy.stats.norm.cdf(-d2)
    
    # Get put price (slightly changed compared to Call)
    V_0  = K * math.exp(-r * (T - t)) * phi_d2  - S_0 * phi_d1    
    
    return phi_d2


# Test function
t = 1
S_0 = 100
r = 0.05
sigma = 0.3
T = 1
M = 4
K = 110

# Print results
S = BlackScholes_EuPut(t, S_0, r, sigma, T, K)
print(S)


# c)

# Set input
t = 1
S_0 = 100
r = 0.05
sigma = 0.3
T = 1
M = range(10, 501, 1)
K = 120

# Set data containers
V_0_EuPut = np.empty(491, dtype=float)
V_0_BS_Put = np.empty(491, dtype=float)

for i in range(0, len(M)):
    V_0_EuPut[i] = CRR_AmEuPut(S_0, r, sigma, T, M[i], K, 1)
    V_0_BS_Put[i] = BlackScholes_EuPut(t, S_0, r, sigma, T, K)


# Plot data
plt.clf()
plt.plot(V_0_EuPut, 'b', label='CRR EuPut')
plt.plot(V_0_BS_Put, 'r', label='BS EuPut')

plt.title('CRR Error against BS-Model')
plt.xlabel('Strike Price (K)')
plt.ylabel('Option Price')

plt.legend()
plt.show()

EuPut = CRR_AmEuPut(S_0, r, sigma, T, M, K, 1)




