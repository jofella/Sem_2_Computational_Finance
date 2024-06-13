# C-Exercise 21, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella

import math
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

#Given BS-formula
# compute Black-Scholes price by integration
def BS_Price_Int(S0, r, sigma, T, f):
    # define integrand as given in the exercise
    def integrand(x):
        return 1 / math.sqrt(2 * math.pi) * f(
            S0 * math.exp((r - 0.5 * math.pow(sigma, 2)) * T + sigma * math.sqrt(T) * x)) * math.exp(-r * T) * math.exp(
            -1 / 2 * math.pow(x, 2))

    # perform integration
    I = integrate.quad(integrand, -np.inf, np.inf)
    # return value of the integration
    return I[0]

#a)
def BS_Greeks_num(r, sigma, S0, T, g, eps):
    Delta = (BS_Price_Int(S0+eps*S0, r, sigma, T, g)-BS_Price_Int(S0, r, sigma, T, g))/(eps*S0)
    vega = (BS_Price_Int(S0, r, sigma+eps*sigma, T, g)-BS_Price_Int(S0, r, sigma, T, g))/(eps*sigma)
    gamma = (BS_Price_Int(S0+eps*S0, r, sigma, T, g)-2*BS_Price_Int(S0, r, sigma, T, g)+BS_Price_Int(S0+eps*S0, r, sigma, T, g))/((eps*S0)**2)

    return [Delta, vega, gamma]


#b)
#define payout function
def payoff(x):
    return np.maximum(x-110, 0)

Deltas = np.empty(0)
for S0 in range(60, 141):
    Deltas = np.append(Deltas, BS_Greeks_num(0.05, 0.3, S0, 1, payoff, 0.001)[0])
plt.clf()
plt.plot(range(60, 141),Deltas)
plt.title("Delta for call option")
plt.xlabel("Initial Stockprice")
plt.ylabel("Delta")
plt.show()


