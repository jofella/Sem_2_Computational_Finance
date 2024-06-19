# C-Exercise 26, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from heston_char import heston_char # from resources



# Heston model function (ref.)
def heston_char(u, S0, r, gam0, kappa, lamb, sig_tild, T):
    ## Compute characteristic function of log-stock price in the Heston model, cf. equation (4.8) with t = 0
    d = np.sqrt(lamb ** 2 + sig_tild ** 2 * (u * 1j + u ** 2))
    phi = np.cosh(0.5 * d * T)
    psi = np.sinh(0.5 * d * T) / d
    first_factor = (np.exp(lamb * T / 2) / (phi + lamb * psi))**(2 * kappa / sig_tild ** 2)
    second_factor = np.exp(-gam0 * (u * 1j + u ** 2) * psi / (phi + lamb * psi))
    return np.exp(u * 1j * (np.log(S0) + r * T)) * first_factor * second_factor



# Power call

def Heston_PCall_Lapl(S0, r, gam0, kappa, lamb, sig_tild, T, K, R, p):
    # Script p. 52, t=0 since we are only interestet in V0 here
    def chi_t(u):
        return heston_char(u - 1j * R, S0, r, gam0, kappa, lamb, sig_tild, T)

    # Script euqation 4.6
    def lap_trans(z):
        return K**(1 - z) / (z * (z - 1)) - S0**(1 - z) / (z * (z - 1))

    def integrand(u):
        return np.real(lap_trans(p + R + 1j * u) * chi_t(u)) #added p here for the power call transformation

    # Perform integration
    int_part, _ = scipy.integrate.quad(integrand, 0, np.inf)

    # Compute option price, again t=0
    V_0 = (np.exp(-r * (T-0)) / np.pi) * int_part
    return V_0



# Test function
S0 = range(50, 151, 1)
r = 0.05
gam0 = 0.32
kappa = 0.32
lamb = 2.5
sig_tild = 0.2
T = 1
K = 100
R = 0  # can be change according to case
p = 1 # power factor



# Compute option prices
option_prices = np.empty(len(S0), dtype=float) 

for i in range(0, len(S0)):
    option_prices[i] = Heston_PCall_Lapl(S0[i], r, gam0, kappa, lamb, sig_tild, T, K, R, p)

print(f'All option prices: {option_prices}')



# plot
plt.clf()

plt.plot(S0, option_prices)
plt.title("Laplace power call prices for different stock prices")
plt.xlabel("Initial Stockprice")
plt.ylabel("Option price")

plt.show()