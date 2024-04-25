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


""""----------------------------------------------------------------
C-Exercise 01
----------------------------------------------------------------"""
# a) CRR Stock movement

# 1.Step: Define function CRR_stock
def CRR_stock(S_0, r, sigma, T, M):
    



# b) CRR European Call



# c) BS European Call


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
dax = np.genfromtxt(r"C:\Users\josef\Documents\GitHub\Sem_2_Computational_Finance\Assignement_01\time_series_dax_2024.csv", 
                    delimiter = ';'
                    , usecols = 4,
                    skip_header = 1)

# 2.Step: Flip ts 
dax = np.flip(dax)

# 3.Step: Apply function to ts
dax_log_returns = log_returns(dax)

# 4.Step: Visualize log returns
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

plt.plot(dax_simulated_log_returns, color='blue')
plt.plot(dax_log_returns, color='red')

plt.title('Comparison Simulated versus observed log returns')
plt.xlabel('Trading Days')
plt.ylabel('Return')

plt.show()


# d) Comparing empirical with simulated data

    
