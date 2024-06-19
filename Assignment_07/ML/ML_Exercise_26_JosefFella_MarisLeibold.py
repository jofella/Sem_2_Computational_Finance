# C-Exercise 26, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella


import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt





def Heston_PCall_Lapl(S0, r, gam0, kappa, lamb, sig_tild, T, K, R, p):
    # from olat material
    def heston_char(u, S0, r, gam0, kappa, lamb, sig_tild, T):
        ## Compute characteristic function of log-stock price in the Heston model, cf. equation (4.8) with t = 0
        d = np.sqrt(lamb ** 2 + sig_tild ** 2 * (u * 1j + u ** 2))
        phi = np.cosh(0.5 * d * T)
        psi = np.sinh(0.5 * d * T) / d
        first_factor = (np.exp(lamb * T / 2) / (phi + lamb * psi)) ** (2 * kappa / sig_tild ** 2)
        second_factor = np.exp(-gam0 * (u * 1j + u ** 2) * psi / (phi + lamb * psi))
        return np.exp(u * 1j * (np.log(S0) + r * T)) * first_factor * second_factor

    #laplace transform of payoff, based on own calculation
    def lap_trans(z):
        return -((K**((p-z)/p)) * (1/(p-z)) + (1/z) * K**(1-(z/p)))

    def integrand(u):
        return np.real(lap_trans(R + 1j * u) * heston_char(u - 1j * R, S0, r, gam0, kappa, lamb, sig_tild, T))

    # Perform integration
    int_part, _ = scipy.integrate.quad(integrand, 0, np.inf, epsabs=1e-2)#increased tolerance so itd work

    # Compute option price, again t=0
    V_0 = (np.exp(-r * (T-0)) / np.pi) * int_part
    return V_0

#plot results for S0=[50:150]
price = np.empty(0)


for i in range(50, 151):
    price = np.append(price, Heston_PCall_Lapl(i, 0.05, 0.09, 0.09, 2.5, 0.2, 1, 100, 5, 1))

#plt.clf()
plt.plot(range(50, 151),price)
plt.title("call option price")
plt.xlabel("Initial Stockprice")
plt.ylabel("Price")
plt.show()

