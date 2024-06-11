# C-Exercise 09, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella



import math
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


# compute Black-Scholes price by integration (given resource)
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

# Test function
S0 = 60
r = 0.05
sigma = 0.3
T = 1
k = 110


# Compute the Black-Scholes price
BS_price = BS_Price_Int(S0, r, sigma, T, f)
print(BS_price)



# a) Compute the goddam greeks


# Vanilla Call-Option function
def g(x):
    return np.maximum(x-k, 0)


def BS_Greeks_num(r, sigma, S0, T, g , eps):
    
    #Option price
    V_BS =  BS_Price_Int(S0, r, sigma, T, g)
    
    # delta, here x = S0
    BS_delta_x_plus_epsilion = BS_Price_Int(S0* (1+eps) , r, sigma, T, g) #getting BS price with eps difference (ref formula sheet)
    delta_BS  = (BS_delta_x_plus_epsilion - V_BS) / (eps * S0)
    
    # vega, here x = sigma
    BS_vega_x_plus_epsilion = BS_Price_Int(S0, r, sigma*(1+eps), T, g)
    vega_BS = (BS_vega_x_plus_epsilion - V_BS) / (eps * sigma)
    
    # gamma, here x = S0
    BS_gamma_x_minus_epsilion = BS_Price_Int(S0*(1-eps), r, sigma, T, g)
    gamma_BS = (BS_delta_x_plus_epsilion - 2 * V_BS + BS_gamma_x_minus_epsilion) / (np.power(eps*S0, 2))
    
    return delta_BS, vega_BS, gamma_BS 

# Set parameters
r = 0.05
sigma = 0.3
T = 1
S0 = 100
k = 110
eps = 0.001


# Test function
delta, vega, gamma = BS_Greeks_num(r, sigma, S0, T, g, eps)
print(delta, vega, gamma)



# b) Visulaize the delta (will lie between 0 and 1, 0.5 = ATM)

# Set parameters
r = 0.05
sigma = 0.3
T = 1
S0 = range(60, 141, 1)
k = 110
eps = 0.001


# Set up data containers
BS_Call_deltas = np.empty(len(S0), dtype=float) # make sure correct datatype

# idea: ref. 06 --> looping thru range and sim. storing correctly
for i in range(0, len(S0)):
    delta, _ , _ = BS_Greeks_num(r, sigma, S0[i], T, g , eps) # we only want delta
    BS_Call_deltas[i] = delta


# plot data
plt.clf()
plt.plot(S0, BS_Call_deltas)

plt.title('Change of delta with different strike prices')
plt.xlabel('strike price (€)')
plt.ylabel('delta')

plt.show()