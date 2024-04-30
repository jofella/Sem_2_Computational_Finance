#Exercise 1
#a
import math
import numpy as np
import matplotlib.pyplot as plt
def CRR_stock(S_0, r, sigma, T, M):
    matrix = np.zeros((M+1, M+1), dtype=float) #set up price matrix in appropriate shape
    delta_t = T / M  # cause all t's have equal distance

    # Skript: 1.4 to 1.7 - Setting u, d, q
    beta = 0.5 * (math.exp(-r * delta_t) + math.exp((r + np.power(sigma, 2)) * delta_t))
    u = beta + np.sqrt(np.power(beta, 2) - 1)
    d = np.power(u, -1)
    q = (math.exp(r * delta_t) - d) / (u - d)
    #print("u is: " + str(u) + " d is: " + str(d))
    t = 0     #first timepoint t_0

    for timesteps in range(0, M+1):
        prices = np.zeros(M + 1)  #set up array to store all possible prices for given t
        for j in range(0,t+1):
            S_ji = S_0 * u**j * d**(t-j)
            prices[j] = S_ji
        matrix[:,t] = prices    #calculate and store all possible prices for given t and paste them in to appropriate column of price matrix
        t += 1
    return matrix


result = CRR_stock(100, 0.03, 0.3, 1, 100)

#1

