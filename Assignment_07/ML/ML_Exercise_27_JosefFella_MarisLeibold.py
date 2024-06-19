# C-Exercise 27, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella


import math
import numpy as np
import scipy.stats
import scipy.optimize
from scipy.interpolate import interp1d

#a)
def ImpVol(V0, S0, r, T, K):
    #create BS call price function
    def price_BS_call(sigma,S0, r, T, K):
        d_1 = (math.log(S0 / K) + (r + 1 / 2 * math.pow(sigma, 2)) * (T)) / (sigma * math.sqrt(T))
        d_2 = d_1 - sigma * math.sqrt(T)
        phi = scipy.stats.norm.cdf(d_1)
        c =  S0 * phi -  K * math.exp(-r * (T)) * scipy.stats.norm.cdf(d_2)
        return c

    #create function to be minimized
    def obj_func(sigma):
        return (price_BS_call(sigma,S0, r, T, K)-V0)**2

    res = scipy.optimize.minimize(obj_func, 0.07, tol=1e-10)
    return res.x[0]

# Test the function with the given values
S0 = 100
r = 0.05
T = 1
K = 100
V0 = 6.09

ImpVol(V0, S0, r, T, K)



#b)
#from material
def heston_char(u, S0, r, gam0, kappa, lamb, sig_tild, T):
    ## Compute characteristic function of log-stock price in the Heston model, cf. equation (4.8) with t = 0
    d = np.sqrt(lamb ** 2 + sig_tild ** 2 * (u * 1j + u ** 2))
    phi = np.cosh(0.5 * d * T)
    psi = np.sinh(0.5 * d * T) / d
    first_factor = (np.exp(lamb * T / 2) / (phi + lamb * psi))**(2 * kappa / sig_tild ** 2)
    second_factor = np.exp(-gam0 * (u * 1j + u ** 2) * psi / (phi + lamb * psi))
    return np.exp(u * 1j * (np.log(S0) + r * T)) * first_factor * second_factor

#from tutorial
def Heston_EuCall(S0, r, gam0, kappa, lamb, sig_tild, T, K, R, N):
    K = np.atleast_1d(K)
    f_tilde_0 = lambda u: 1 / (u * (u - 1))
    chi_0 = lambda u: heston_char(u, S0=S0, r=r, gam0=gam0, kappa=kappa, lamb=lamb, sig_tild=sig_tild, T=T)
    g = lambda u: f_tilde_0(R + 1j * u) * chi_0(u - 1j * R)

    kappa_1 = np.log(K[0])
    M = np.minimum(2 * np.pi * (N - 1) / (np.log(K[-1] - kappa_1)), 500)
    Delta = M / N
    n = np.arange(1, N + 1)
    kappa_m = np.linspace(kappa_1, kappa_1 + 2 * np.pi * (N - 1) / M, N)

    x = g((n - 0.5) * Delta) * Delta * np.exp(-1j * (n - 1) * Delta * kappa_1)
    x_hat = np.fft.fft(x)

    V_kappa_m = np.exp(-r * T + (1 - R) * kappa_m) / np.pi * np.real(x_hat * np.exp(-0.5 * Delta * kappa_m * 1j))
    return interp1d(kappa_m, V_kappa_m)(np.log(K))


def ImpVol_Heston( S0, r, gam0, kappa, lamb, sig_tild, T, K, R):
    #calculate call option price per instructions
    price = Heston_EuCall(S0, r, gam0, kappa, lamb, sig_tild, T, K, R, 2**15)

    # create function to be minimized
    def obj_func(sig_tild):
        return (Heston_EuCall(S0, r, gam0, kappa, lamb, sig_tild, T, K, R, 2**15) - price) ** 2

    res = scipy.optimize.minimize(obj_func, 0.07, tol=1e-10)
    return res.x[0]



vol = np.empty(0)


for i in range(50, 70):
    vol = np.append(vol, ImpVol_Heston(i, 0.05, 0.09, 0.9, 2.5, 0.2, 1, 100, 5))
    print(i)