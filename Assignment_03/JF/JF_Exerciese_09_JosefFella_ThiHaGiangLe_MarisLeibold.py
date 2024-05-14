# C-Exercise 09, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella, Thi Ha Giang Le

import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


# Fix random seed
np.random.seed(28) # my favorite number


def Sample_TruncNormal_AR(a, b, mu, sigma, N):
    
    # 1.Step: Generate uniform samples size N
    U_1 = np.random.uniform(a, b, N)
    
    # 2.Step: Define traget distribution (truncated normal)
    transformed_normal_pdf  = scipy.stats.norm.pdf((U_1 - mu) / sigma)
    transformed_normal_cdf_b = scipy.stats.norm.cdf((b - mu) / sigma)
    transformed_normal_cdf_a = scipy.stats.norm.cdf((a - mu) / sigma)
    
    trunc_normal_sample = transformed_normal_pdf / (sigma * (transformed_normal_cdf_b - transformed_normal_cdf_a))

    # 3.Step: Generate second uniform sample as checking sample
    U_2 = np.random.uniform(a, b, N)
    
    
    # 4.Step: Checking if U =< f(Y) / c*g(Y) else go back
    if U_2 <= (trunc_normal_sample / U_1):
        print("You are right")
    
    else:
        return ("You are wrong")
    
    
# Testing

a = 0
b = 1
mu = 0
sigma = 1
N = 100


# 1.Step: Generate uniform samples size N
U_1 = np.random.uniform(a, b, N)

    
# 2.Step: Define traget distribution (truncated normal)
transformed_normal_pdf  = scipy.stats.norm.pdf((U_1 - mu) / sigma)
transformed_normal_cdf_b = scipy.stats.norm.cdf((b - mu) / sigma)
transformed_normal_cdf_a = scipy.stats.norm.cdf((a - mu) / sigma)

trunc_normal_sample = transformed_normal_pdf / (sigma * (transformed_normal_cdf_b - transformed_normal_cdf_a))


# 3.Step: Generate second uniform sample as checking sample
U_2 = np.random.uniform(a, b, N)

# Check ratio
ratio = trunc_normal_sample / U_1


# 4.Step: Checking if U =< f(Y) / c*g(Y) else go back
if U_2 <= ratio:
    print("You are right")
    
else:
    print("You are wrong")



####################################################


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
plt.hist(pdf_standard_normal)

plt.title(f'Histogram of normal distribution with $\mu = {mu}$ and $\sigma = {sigma}$') # from matplotlib doc
plt.xlabel('bins')

plt.show()