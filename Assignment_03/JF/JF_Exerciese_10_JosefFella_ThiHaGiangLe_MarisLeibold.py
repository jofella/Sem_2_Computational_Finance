# C-Exercise 10, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella, Thi Ha Giang Le

import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


# Fix random seed
np.random.seed(28) # my favorite number


# Build function

def BS_EuOption_MC_CV(S_0, r, sigma, T, K, M):
    



# Test function
S_0 = 100
r = 0.05
sigma = 0.3
T = 1
K = 110
M = 100000

V0 = BS_EuOption_MC_CV(S0, r, sigma, T, K, M)
    


# MC from 07 --- Copy from solution
def Eu_Option_BS_MC(S_0, r, sigma, T, K, M, f):
    # generate M samples
    X = np.random.normal(0, 1, M)
    ST = np.empty(len(X), dtype=float)
    Y = np.empty(len(X), dtype=float)

    # compute ST and Y for each sample
    for i in range(0, len(X)):
        ST[i] = S0 * math.exp((r - 0.5 * math.pow(sigma, 2)) * T + sigma * math.sqrt(T) * X[i])
        Y[i] = f(ST[i], K)

    # calculate V0
    VN_hat = np.mean(Y)
    # calculate V0
    V0 = math.exp(-r * T) * VN_hat

    # compute confidence interval
    epsilon = 1.96 * math.sqrt(np.var(Y) / M)
    c1 = math.exp(-r * T) * (VN_hat - epsilon)
    c2 = math.exp(-r * T) * (VN_hat + epsilon)
    return V0, VN_hat, c1, c2



# Comparing the confidence intervals (since variance reduction method)
