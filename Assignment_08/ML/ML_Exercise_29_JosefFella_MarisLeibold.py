# C-Exercise 29, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella

import numpy as np
from scipy.stats import norm
import scipy.misc


#Define payoff function g
def g(St):
    return np.maximum(St-90, 0)



def EuOptionHedge_BS_MC_IP (St, sigma, g, T, t, N):
    X = np.random.normal(0, 1, N)
    Zn = np.empty(len(X))
    ST = np.empty(len(X))

    #Define function that corresponds to Z(v) in script with mean(Zn) being the estimate for z(v)
    def fun(St):
        ST = St *np.exp(((r-sigma**2)/2)*(T-t) + sigma *((T-t)**0.5)*X)
        Zn = g(ST)
        return np.mean(Zn)

    #differentiate above function w.r.t. S(t)
    z = scipy.misc.derivative(fun, St)
    return z





#Define testfunction according to script
def phi_t (St, K, r, T, t, sigma):
    num = np.log(St/K) + r*(T-t) + ((sigma**2)/2)*(T-t)
    denom = sigma * ((T-t)**0.5)
    return norm.cdf(num/denom)

t = 0
T = 1
St = 100
r = 0.05
sigma = 0.2
K =90
N = 100000


phil_t_exact = phi_t (St, K, r, T, t, sigma)
phil_t =EuOptionHedge_BS_MC_IP(St, sigma , g , T, t, N)

print("Exact phi_1: " , phil_t_exact ,
      "Approximated phi_1: " , phil_t)

