# C-Exercise 23, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella


import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt


def BS_EuCall_Laplace(S0, r, sigma, T, K, R):
    # Script p. 52, t=0 since we are only interestet in V0 here
    def chi_t(u):
        return np.exp(1j * u * (np.log(S0) + r * (T-0)) - (1j * u + u ** 2) * (sigma ** 2 / 2) * (T-0))

    # Script euqation 4.6
    def lap_trans(z):
        return K ** (1 - z) / (z * (z - 1))

    def integrand(u):
        return np.real(lap_trans(R + 1j * u) * chi_t(u - 1j * R))

    # Perform integration
    int_part, _ = scipy.integrate.quad(integrand, 0, np.inf)

    # Compute option price, again t=0
    V_0 = (np.exp(-r * (T-0)) / np.pi) * int_part
    return V_0




#plot results for S0=[50:150]
price = np.empty(0)

for i in range(50, 151):
    price = np.append(price, BS_EuCall_Laplace(i, 0.03, 0.2, 1, 110, 10))

plt.clf()
plt.plot(range(50, 151),price)
plt.title("call option price")
plt.xlabel("Initial Stockprice")
plt.ylabel("Price")
plt.show()

