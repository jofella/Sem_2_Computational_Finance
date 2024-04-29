""""----------------------------------------------------------------
Group Information 
----------------------------------------------------------------"""

# Group name: QF Group 02

# Members:
# Josef Fella
# Thi Ha Giang Le
# Maris Leibold


""""----------------------------------------------------------------
General Settings
----------------------------------------------------------------"""
# importing packages

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

""""----------------------------------------------------------------
C-Exercise 01
----------------------------------------------------------------"""
# a) CRR Stock movement

# 1.Step: Define function CRR_stock

# Idea: We have a non-recombining tree so we just care about the results. 
# Basically applied results from skript 1.5. Algorithm and discussion.

def CRR_stock(S_0, r, sigma, T, M):
    
    delta_t = T/M #cause all t's have equal distance
    
    # Skript: 1.4 to 1.7 - Setting u, d, q
    beta = 0.5 * (math.exp(-r * delta_t) + math.exp(((r + np.power(sigma, 2)) * delta_t)))
    u = beta + np.sqrt(np.power(beta, 2) - 1)
    d = np.power(u, -1)
    q = (math.exp(r * delta_t) - d) / (u - d)

    # Set up empty S_ji array, Reminder:  M cause python index start at 0
    S_ji = np.empty((M + 1, 1))
    
    # Apply our CRR_Formula S_ji = S_0 * u^j * d^(i-j) 
    for j in range(M + 1):
        S_ji[j] = S_0 * np.power(u, j) * np.power(d, M - j)
    
    S_ji = np.flip(S_ji)
    return (S_ji)

# Test function
S = CRR_stock(100, 0.03, 0.3, 1, 100) #Values from d)

# Sense check S[50] == S_0?
print(S[100])


M = 100
# Testing
S_ji = np.empty((M + 1),(M + 1) )
print(np.where(S_ji))




# b) CRR European Call

def CRR_EuCall(S_0, r, sigma, T, M, K):
    
    return V_0


# c) BS European Call
def BlackScholes_EuCall(t, S_t, r, sigma, T, K):
    d1 = (np.log(S_t/K) + (r + (np.power(sigma, 2)/2)) * (T-t)) / (sigma * np.sqrt(T-t))
    d2 = d1 - sigma * np.sqrt(T-t)
    
    #Imported function from scipy to get normal cdf value
    phi_d1 = norm.cdf(d1)
    phi_d2 = norm.cdf(d2)
    
    V_0  = S_t * phi_d1 - K * math.exp(-r * (T-t)) * phi_d2    
    
    return V_0

V_0_BS = BlackScholes_EuCall(0, 100, 0.03, 0.3, 100, 70)
print(V_0_BS)



# d) Comparing BS and CRR






""""----------------------------------------------------------------
T-Exercise 02
----------------------------------------------------------------"""

# a) Function for log returns

# Idea: Looping thru the ts/array and calculate the log returns according to the formula

def log_returns(data):
    log_returns = np.diff( np.log(data))
    return log_returns


# b) Test the function

## Part 1: Working with log-returns

# 1.Step: Import data
dax = np.genfromtxt("time_series_dax_2024.csv", 
                    delimiter = ';'
                    , usecols = 4,
                    skip_header = 1)



# 2.Step: Flip ts 
dax = np.flip(dax)

# 3.Step: Apply function to ts
dax_log_returns = log_returns(dax)

# 4.Step: Visualize log returns

plt.clf()
plt.plot(dax_log_returns)

plt.title('DAX Log-Returns (1990-2024)')
plt.xlabel('Trading days')
plt.ylabel('Daily Log-Return')

plt.show()


## Part 2: Annualized empirical mean and standard deviation

# 1.Step: Get empirical mean

# idea: Define all parts of formula and then calculate it
trading_days = 250
N = len(dax_log_returns)
sum_lk = 0

for k in range(N - 2):
    sum_lk += dax_log_returns[k + 2]
        
# Calculate empirical mu
dax_empirical_mu = trading_days / (N - 1) * sum_lk

print(dax_empirical_mu)


# 2.Step: Get empirical variance
trading_days = 250
N = len(dax_log_returns)
sum_lk = 0

for k in range(N - 2):
    sum_lk += np.power(dax_log_returns[k + 2] - dax_empirical_mu / trading_days, 2)

print(sum_lk)

# Calculate empirical sigma
dax_empirical_sigma = trading_days / (N - 2) * sum_lk

print(dax_empirical_sigma)


# c) Simulate ts of log returns with normal distribution

# 1.Step: Generate normally distributed values with our imput parameters
dax_simulated_log_returns = np.random.normal(dax_empirical_mu, dax_empirical_mu, N)
print(dax_simulated_log_returns)


# 2.Step: Plot results
plt.clf()
plt.plot(dax_simulated_log_returns, color='blue')
plt.plot(dax_log_returns, color='red')

plt.title('Comparison Simulated versus observed log returns')
plt.xlabel('Trading Days')
plt.ylabel('Return')

plt.show()


# d) Comparing empirical with simulated data

# The simulated are way more volatile than the actual market log returns. 
# --> Normal distribution are related to "fat tails" so here we are overestimating our risk ???
# seems weird tbh