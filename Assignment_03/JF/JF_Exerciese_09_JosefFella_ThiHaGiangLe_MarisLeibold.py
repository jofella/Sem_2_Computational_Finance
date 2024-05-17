# C-Exercise 09, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella, Thi Ha Giang Le

import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# Fix random seed
np.random.seed(28)


# Set sample generator
def Sample_TruncNormal_AR(a, b, mu, sigma, N):
    
    # 1.Step: Set up container & loop counter
    trunc_normal_sample = np.empty(N, dtype=float)
    i = 0
    c = 1 #must be calculated?
    
    while i < N:
        # Generate iid uniform samples
        y_sample = np.random.uniform(a, b, 1)
        u_sample = np.random.uniform(0, 1, 1)
        
        # Simulate pdfs: f_y trunc normal pdf and g_y = Uniform pdf 
        f_y = scipy.stats.norm.pdf((y_sample - mu) / sigma) / (sigma * (scipy.stats.norm.cdf((b - mu)/sigma) - scipy.stats.norm.cdf((a - mu)/sigma)))
        g_y = scipy.stats.uniform.pdf(y_sample, a, b-a) #why b-a?
        
        ratio = f_y / (c * g_y)

        # Accept, else reject (remember its about the pdf not the sample itself)
        if u_sample <= ratio:
            trunc_normal_sample[i] = y_sample
            i += 1

    return trunc_normal_sample


# Testing
a = 0
b = 2
mu = 0.5
sigma = 1
N = 100000 #sample size


Test = Sample_TruncNormal_AR(a, b, mu, sigma, N)
print(Test)

plt.clf()
plt.hist(Test, density=True)
x_range = np.linspace(-1,3,1000)
plt.plot(x_range, scipy.stats.truncnorm.pdf(x_range, (a-mu)/sigma, (b-mu)/sigma, mu, sigma), "r")
plt.show()



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


plt.show()