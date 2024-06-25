# C-Exercise 28, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella

import numpy as np
import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

### Adjust data path if necessary
data = np.genfromtxt('option_prices_sp500', delimiter=',', skip_header=1)



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

def calibrate_Heston(V0_data, S0, r, K, T):
    def f(theta):
        gam0, kappa, lamb, sig_tild = theta
        V0 = np.empty(128)
        for i in range(0,128):
            V0[i] = Heston_EuCall(S0[i], r[i], gam0, kappa, lamb, sig_tild, T, K[i], 5, 2 ** 10)

        return np.sum((V0-V0_data)**2)

    res =scipy.optimize.minimize(f, [1, 1, 1, 1])
    return res.x


#retrive calibration data from provided data
V0_data = data[:,1]
r = data[:,3]
S0 = data[:,2]
K = data[:, 0]
T = 1

theta = calibrate_Heston(V0_data, S0, r, K, T)


#plotting
hes_price = np.empty(128)
for i in range(0,128):
    hes_price[i] = Heston_EuCall(S0[i], r[i], theta[0]**0.5, theta[1]**0.5, theta[2], theta[3], T, K[i], 10, 2 ** 15)

plt.clf()
plt.plot(K,V0_data)
plt.plot(K,hes_price)
plt.ylabel('Option Price')
plt.xlabel('Strike Price')


