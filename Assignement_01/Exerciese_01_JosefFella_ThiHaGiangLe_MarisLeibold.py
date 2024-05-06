""""----------------------------------------------------------------
Group Information 
----------------------------------------------------------------"""

# Group name: QF Group 02

# Member:
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

# Idea: We have a recombining tree, so in a) we apply the first part 
# of 1.5. Algorithm and discussion

def CRR_stock(S_0, r, sigma, T, M):
    
    delta_t = T/M
    
    # Skript: 1.4 to 1.7 - Setting u, d, q
    beta = 0.5 * (math.exp(-r * delta_t) + math.exp(((r + np.power(sigma, 2)) * delta_t)))
    u = beta + np.sqrt(np.power(beta, 2) - 1)
    d = 1 / u
    q = (math.exp(r * delta_t) - d) / (u - d)

    # Set up empty S_ji array, Reminder:  M cause python index start at 0
    S_ji = np.zeros((M + 1, M + 1))
    
    # Apply our CRR_Formula S_ji = S_0 * u^j * d^(i-j)
    
    for i in range(M + 1):
        for j in range(i + 1):
            S_ji[i, j] = S_0 * np.power(u, j) * np.power(d, i - j)
    
    return (S_ji)


# Test function

S_0 = 100
r = 0.03
sigma = 0.3
T = 1
M = 100


# Print results:
S = CRR_stock(S_0, r, sigma, T, M) #Values from d)
print(S)



# b) CRR European Call

def CRR_EuCall(S_0, r, sigma, T, M, K):
    
    ## Part a) -- Generally we could also use our CRR_stock function for first part
    delta_t = T/M
    
    # Skript: 1.4 to 1.7 - Setting u, d, q
    beta = 0.5 * (math.exp(-r * delta_t) + math.exp(((r + np.power(sigma, 2)) * delta_t)))
    u = beta + np.sqrt(np.power(beta, 2) - 1)
    d = 1 / u
    q = (math.exp(r * delta_t) - d) / (u - d)

    # Set up empty S_ji array, Reminder:  M cause python index start at 0
    S_ji = np.zeros((M + 1, M + 1))
    
    # Apply our CRR_Formula S_ji = S_0 * u^j * d^(i-j)
    
    for i in range(M + 1):
        for j in range(i + 1):
            S_ji[i, j] = S_0 * np.power(u, j) * np.power(d, i - j)
    
    ## New Part:
    
    # Compute discounted Euro Call-Payoff in M According to 1.16: 
    S_ji[-1,:] = np.maximum(S_ji[-1,:] - K, 0) / math.exp(M*r)
    
    # Get discounted Values for portfolio & Looping backwards
    V_ji = S_ji
    
    for i in range(M, 0, -1):
        for j in range(i):
            V_ji[i-1, j] = math.exp(-r * delta_t) * (q * V_ji[i, j]+ (1-q) * V_ji[i, j+1])
    
    # Undo Discounting
    V_ji = V_ji * math.exp(-r * delta_t)
    V_0 = V_ji[0,0]
    
    return V_0


# Test function
S_0 = 100
r = 0.03
sigma = 0.3
T = 1
M = 4
K = 110 # the closer ITM (in the money) the higher the price


S = CRR_EuCall(S_0, r, sigma, T, M, K)
print(S)



# c) BS European Call
def BlackScholes_EuCall(t, S_t, r, sigma, T, K):
    d1 = (np.log(S_t/K) + (r + (np.power(sigma, 2)/2)) * (T-t)) / (sigma * np.sqrt(T-t))
    d2 = d1 - sigma * np.sqrt(T-t)
    
    # Imported from scipy to get normal cdf value
    phi_d1 = norm.cdf(d1)
    phi_d2 = norm.cdf(d2) #double check values with numpy
    
    V_0  = S_t * phi_d1 - K * math.exp(-r * (T-t)) * phi_d2    
    
    return V_0


# Test function
t = 0
S_0 = 100
r = 0.03
sigma = 0.3
T = 1
K = 130 # the closer ITM (in the money) the higher the price


V_0_BS = BlackScholes_EuCall(t, S_0, r, sigma, T, K)
print(V_0_BS)



# d) Comparing BS and CRR

# Set up Parameters
S_0 = 100
r = 0.03
sigma = 0.3
t = 0
T = 1
M = 100
K = range(70, 201)
print(K)

## Calculate option prices

# 1.CRR-model
CRR_prices = np.zeros(len(K))
# For using only one index we have i-K[0]
for i in K:
    CRR_prices[i - K[0]] = CRR_EuCall(S_0, r, sigma, T, M, i)


# 2.BS-model
BS_prices = np.zeros(len(K))
for i in K:
    BS_prices[i - K[0]] = BlackScholes_EuCall(t, S_0, r, sigma, T, i)


# Plot results
plt.clf()
plt.plot(CRR_prices, 'b', label='CRR Prices')
plt.plot(BS_prices, 'r', label='BS Prices')

plt.title('CRR Error against BS-Model')
plt.xlabel('Strike Price (K)')
plt.ylabel('Option Price')

plt.legend()
plt.show()



""""----------------------------------------------------------------
T-Exercise 02
----------------------------------------------------------------"""

# a) Function for log returns

# Idea: Looping thru the ts/array and calculate the log returns according to the formula

def log_returns(data):
    return np.diff( np.log(data))


# b) Test the function

## Part 1: Working with log-returns

# 1.Step: Import data

# REMARK: We saw your comment on not including the path. But even when we put the file
# in the same folder it doesn't work for all of us. 
# We stored the path now in a seperate variable now to make it easier to change.

path = r"C:\Users\josef\Documents\GitHub\Sem_2_Computational_Finance\Assignement_01\time_series_dax_2024.csv"
dax = np.genfromtxt(path, 
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
plt.ylabel('Daily Log-Returns')

plt.show()


## Part 2: Annualized empirical mean and standard deviation

# 1.Step: Get empirical mean

# idea: Define all parameters of formula and then calculate it
trading_days = 250
N = len(dax_log_returns)
sum_lk = 0

for k in range(N - 1):
    sum_lk += dax_log_returns[k + 1]
       
# Calculate empirical mu
dax_empirical_mu = trading_days / (N - 1) * sum_lk

print(dax_empirical_mu)

# 2.Step: Get empirical variance
trading_days = 250
N = len(dax_log_returns)
sum_lk = 0

for k in range(N - 1):
    sum_lk += np.power(dax_log_returns[k + 1] - dax_empirical_mu / trading_days, 2)

print(sum_lk)

# Calculate empirical sigma
dax_empirical_sigma = np.sqrt(trading_days / (N - 2) * sum_lk)

print(dax_empirical_sigma)


# c) Simulate ts of log returns with normal distribution

# 1.Step: Generate normally distributed values with our imput parameters
dax_simulated_log_returns = np.random.normal(dax_empirical_mu, dax_empirical_sigma, N)
print(np.mean(dax_log_returns), np.std(dax_log_returns))

# 2.Step: Plot results
plt.clf()
plt.plot(dax_simulated_log_returns, color='blue', label='Simulated Log-Returns')
plt.plot(dax_log_returns, color='red', label='Observed Log-Returns')

plt.title('Comparison Simulated versus observed log returns')
plt.xlabel('Trading Days')
plt.ylabel('Return')

plt.legend()
plt.show()


# d) Comparing empirical with simulated data

# The simulated log returns are way higher than the actual observed ones. This suggests
# that the normal distribution may not be a good approximation for real world market data.
# So in general it seems that the overall range of returns for our normal data is higher.

# Further the simulated returns look much more homogenous. When you compare it to the observed ones
# you see here different "clusters" meaning phases with large volatility and phases with very low vola.
# This is not shown by our simulated returns.

# This also referce to the so called "fat tails". So the problem with using normal distribution to simulate market data
# is, that generaly the the extreme values (tails) are underestimated by the normal. Hence, it indicated that its not a
# perfect "estimator".