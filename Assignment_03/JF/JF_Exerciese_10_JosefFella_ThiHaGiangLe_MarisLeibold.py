# C-Exercise 10, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella, Thi Ha Giang Le

import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


# Fix random seed
np.random.seed(28)


# 1.Part: Set option payout functions
def eu_call_payoff(x, K):
    return np.maximum(x - K, 0) # more dynamic with K input function

def quanto_call_payoff(S_T, K):
    return np.maximum(S_T - K, 0) * S_T


# 2.Part: Set "Normal" MC price -- Corrected & adapted version of 07 --
np.random.seed(28)

def Eu_Option_BS_MC(S_0, r, sigma, T, K, M, f):
   
   # 1.Step: Set up data
   X = np.random.normal(0, 1, M)
   S_T = np.empty(len(X), dtype = float)
   Y = np.empty(len(X), dtype = float)
   
   # 2.Step: Simulate stock price with BS
   for i in range(0, M, 1):
        S_T[i] = S_0 * math.exp((r - 0.5 * np.power(sigma, 2)) * T + sigma * np.sqrt(T) * X[i]) # forgot 0.5
        Y[i] = f(S_T[i], K) #going thru all stock prices, otherwise constant!!!
        
   # 3.Step: Calculate V_hat (estimator) and V_0
   V_hat = np.mean(Y) # Estimator
   V_0 = math.exp(-r * T) * V_hat
   
   # 4.Step: Calculate 95% CI (skript p.13) -- corrected due to mistake
   epsilon = 1.96 * math.sqrt(np.var(Y) / M) # ref. solutions
   
   c1 = math.exp(-r * T) * (V_hat - epsilon)
   c2 = math.exp(-r * T) * (V_hat + epsilon)

   return V_0, c1, c2

# Test function
S_0 = 100
r = 0.05
sigma = 0.3
T = 1
K = 110
M = 100


V_0, V_hat, c1, c2 = Eu_Option_BS_MC(S_0, r, sigma, T, K, M, quanto_call_payoff) # first forgot K
print(V_0, V_hat, c1, c2)



# 3. Part: Apply Variance reduction
np.random.seed(28)

def BS_EuOption_MC_CV(S_0, r, sigma, T, K, M):
    
    # 1.Step: Define option price functions (here easier leave inside)
    def quanto_call_payoff(S_T, K):
        return np.maximum(S_T - K, 0) * S_T
    
    def eu_call_payoff(S_T, K): # As control variate: Close to original payoff (+correlation)
        return np.maximum(S_T - K, 0)


    # 2.Step: Set up data
    X_normal_sample = np.random.normal(0, 1, M)
    S_T = np.empty(len(X_normal_sample), dtype = float)
    
    X_payoff = np.empty(len(X_normal_sample), dtype = float)
    Y_payoff = np.empty(len(X_normal_sample), dtype = float)
    
    
    # 3.Step: Simulate prices by BS at T
    for i in range(0, M, 1):
        S_T[i] = S_0 * math.exp((r - 0.5 * np.power(sigma, 2)) * T + sigma * np.sqrt(T) * X_normal_sample[i]) # forgot 0.5
        X_payoff[i] = quanto_call_payoff(S_T[i], K)
        Y_payoff[i] = eu_call_payoff(S_T[i], K) 
    
    
    # 4.Step: Estimate beta -- np.cov retruns a matrix --> either [0,1] or [1,0]
    beta_hat = np.cov(X_payoff, Y_payoff)[0,1] / np.var(Y_payoff)    

    # 5.Step: Get E(Y) - Essentially the BS price for our EuCall (we know it 2.10)    
    E_Y = np.mean(Y_payoff)
    
    # 6. Step: Calculate estimates and option price 
    V_hat_CV_beta = np.mean(X_payoff - Y_payoff * beta_hat + beta_hat * E_Y)      
    V_0 = math.exp(-r * T) * V_hat_CV_beta
    
    
    # 7. Step: Calculate 95%-CI
    var_hat_CV = np.var(X_payoff) + 1/M * (np.var(Y_payoff * beta_hat - 2 * np.cov(X_payoff, Y_payoff)[0,1] ))
    epsilon = 1.96 * math.sqrt(var_hat_CV / M) # ref. solutions
   
    c1 = math.exp(-r * T) * (V_hat_CV_beta - epsilon)
    c2 = math.exp(-r * T) * (V_hat_CV_beta + epsilon)

    return V_0, c1, c2


# Test function
S_0 = 100
r = 0.05
sigma = 0.3
T = 1
K = 110
M = 10000


test, c1, c2 = BS_EuOption_MC_CV(S_0, r, sigma, T, K, M)
print(test, c1, c2)



# 3. Part: Comparing the confidence intervals (since variance reduction method)

# parameters
S_0 = 100
r = 0.05
sigma = 0.3
T = 1
K = 110
M = 10000


# Set up output data
V_0_vanilla_MC, c1_vanilla_MC, c2_vanilla_MC = Eu_Option_BS_MC(S_0, r, sigma, T, K, M, quanto_call_payoff)
V_0_CV_MC, c1_CV_MC, c2_CV_MC = BS_EuOption_MC_CV(S_0, r, sigma, T, K, M)


# Output results
print(f"For vanilla MC we get V0: {V_0_vanilla_MC} and CI: {c1_vanilla_MC} to {c2_vanilla_MC}")
print(f"For variance reduction (CV) MC we get V0: {V_0_CV_MC} and CI: {c1_CV_MC} to {c2_CV_MC}")
print(f"The difference is {c2_CV_MC - c1_CV_MC} to {c2_vanilla_MC - c1_vanilla_MC}")

# Result should be that its tighter, but the price the same ...