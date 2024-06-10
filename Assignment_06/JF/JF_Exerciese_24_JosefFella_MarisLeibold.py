# C-Exercise 09, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella



import math
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt



# Option paypoff
def g(x):
    return np.maximum(k-x)


def BS_AmPerpPut_ODE(S_max, N, r, sigma, K):
    
    
    np.power(sigma, 2) / 2
    
    
    return



# Set parameters
S_max = 200
N = 200
r = 0.05
k = 100
sigma = np.sqrt(0.4) # sigma^2
K = 100


# Test function
V0 = BS_AmPerpPut_ODE(S_max, N, r, sigma, K)


# Plot data
plt.clf()
plt.plot()

plt.title('Change of delta with different strike prices')
plt.xlabel('strike price (€)')
plt.ylabel('delta')

plt.show()