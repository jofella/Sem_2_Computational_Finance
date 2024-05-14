# C-Exercise 09, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella, Thi Ha Giang Le

import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


# Fix random seed
np.random.seed(28) # my favorite number


# Set target function -- here TruncNormal
def target_function(a, b, mu, sigma, N):
        # Uniform sample over range a and b
        U_1 = np.random.uniform(a, b, N)
        
        # Calculating parts of f(X) -- ref. Sheet
        transformed_normal_pdf  = scipy.stats.norm.pdf((U_1 - mu) / sigma)
        transformed_normal_cdf_b = scipy.stats.norm.cdf((b - mu) / sigma)
        transformed_normal_cdf_a = scipy.stats.norm.cdf((a - mu) / sigma)
        
        g_x = transformed_normal_pdf / (sigma * (transformed_normal_cdf_b - transformed_normal_cdf_a))

        return U_1, g_x


# Set sample generator
def Sample_TruncNormal_AR(a, b, mu, sigma, N):
    
    # 1.Step: Set up container & loop counter
    trunc_normal_sample = np.empty(N, dtype=float)
    i = 0
    c = 9999
    max_iter = 10000 # safety mechanism
    
    while i < N and max_iter > 0:
        # Get samples (2 uniform + target function)
        U_1, g_x = target_function(a, b, mu, sigma, 1)
        U_2 = np.random.uniform(0, 1, 1)
        
        ratio = g_x / c * U_1
        
        # Accept
        if U_2 <= ratio:
            trunc_normal_sample[i] = U_2
            i += 1
        
        # Reject - basically doing the same but not adding it -- maybe adding a safety mechanism
        max_iter -= 1

    return trunc_normal_sample


# Testing
a = 0
b = 1
mu = 0
sigma = 1
N = 10 #sample size


Test = Sample_TruncNormal_AR(a, b, mu, sigma, N)
print(Test)




########################################################################################################

# Fix parameters
a = 0
b = 2
mu = 0.5
sigma = 1
N = 10000

TruncNormal_sample = Sample_TruncNormal_AR(a,b, mu, sigma, N)
print(TruncNormal_sample)


# Visualize data in histogram + f(x)
plt.clf()
plt.hist(TruncNormal_sample)

plt.title(f'Histogram of normal distribution with $\mu = {mu}$ and $\sigma = {sigma}$') # from matplotlib doc
plt.xlabel('bins')

plt.show()