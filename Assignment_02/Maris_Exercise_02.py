""""----------------------------------------------------------------
Group Information
----------------------------------------------------------------"""

# Group name: QF Group 02

# Member:
# Josef Fella
# Thi Ha Giang Le
# Maris Leibold

import numpy as np
import math
import scipy.stats
import matplotlib.pyplot as plt

#1a
#includes borrowed parts from the solutions for sheet 01
def CRR_stock(S_0, r, sigma, T, M):
    # compute values of u, d and q
    delta_t = T / M
    alpha = math.exp(r * delta_t)
    beta = 1 / 2 * (1 / alpha + alpha * math.exp(math.pow(sigma, 2) * delta_t))
    u = beta + math.sqrt(math.pow(beta, 2) - 1)
    d = 1 / u

    # allocate matrix S
    S = np.empty((M + 1, M + 1))

    # fill matrix S with stock prices
    for i in range(1, M + 2, 1):
        for j in range(1, i + 1, 1):
            S[j - 1, i - 1] = S_0 * math.pow(u, j - 1) * math.pow(d, i - j)

    return S, u, d
def CRR_AmEuPur(S_0, r, sigma, T, M, K, EU):
    # use function from part a to compute S, u and d
    S, u, d = CRR_stock(S_0, r, sigma, T, M)
    delta_t = T / M
    q = (math.exp(r * delta_t) - d) / (u - d)

    # V will contain the put prices
    V = np.empty((M + 1, M + 1))
    # compute the prices of the put for all times t_i
    for m in range(0, M+1):
        V[:, m] = np.maximum(K - S[:, m], 0)
    #replace the value 1.2 in V, these are nor real
    V[V == 1.2] = 0

    #define recursion function
    def g(k):
        return math.exp(-r * delta_t) * (q * V[1:k + 1, k] + (1 - q) * V[0:k, k])


    if EU == 1: #in case of European Option

        # compute call prices at t_i
        for k in range(M, 0, -1):
            V[0:k, k - 1] = g(k)

        # return the price of the call at time t_0 = 0
        return V[0,0]

    elif EU == 0: #in case of American Option

        for k in range(M, 0, -1):
            put_price = g(k)
            exercise_price = V[k-1]
            exercise_price = exercise_price[0:len(put_price)]
            V[0:k, k - 1] = np.maximum(exercise_price, put_price)

        return V[0,0]

#check answers for 1
V_0 =CRR_AmEuPur(1, 0.05, 0.3**0.5, 3, 3, 1.2, 0)


#1b
def BlackScholes_EuPut (t, S_0, r, sigma, T, K):
    d_1 = (math.log(S_0 / K) + (r + 1 / 2 * math.pow(sigma, 2)) *
           (T - t)) / (sigma * math.sqrt(T - t))
    d_2 = d_1 - sigma * math.sqrt(T - t)
    phi = scipy.stats.norm.cdf(-d_1)
    P =  K * math.exp(-r * (T - t)) * scipy.stats.norm.cdf(-d_2) - S_0 * phi
    return P

#1c
S_0 = 100
r = 0.05
sigma = 0.3
T = 1
M = range(10, 501)
K = 120
n=BlackScholes_EuPut(0, S_0, r, sigma, T, K)
V_0 = np.empty(491, dtype=float)
V_0_BS = np.empty(491, dtype=float)

for m in range(0, len(M)):
    V_0[m] = CRR_AmEuPur(S_0, r, sigma, T, M[m], K, 1)
    V_0_BS[m] = BlackScholes_EuPut(0, S_0, r, sigma, T, K)

plt.clf()
plt.plot(V_0, "r")
plt.plot(V_0_BS, "b")
plt.show()

print(CRR_AmEuPur(S_0, r, sigma, T, 500, K, 0))



###Ex2

def EU_Option_BS_MC(S0, r, sigma, T, K, M, f):
    #Generate BS EU Call Option price for t=0
    d_1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) *T) / (sigma * T**0.5)
    d_2 = d_1 - sigma * T**0.5
    phi = scipy.stats.norm.cdf(d_1)
    V0_BS = S0 * phi - K * math.exp(-r * T) * scipy.stats.norm.cdf(d_2)
    #Setup for MC
    X = np.random.normal(0, 1, size=M)
    S_T = S0*np.exp((r - sigma**2/2)*T+sigma*T**0.5*X)
    #Calculate payoff with provided function
    V0_MC = math.exp(-r*T)*np.mean(f(S_T, K))
    #95% confidence interval
    s_var = np.var(S_T)
    CI95 = [V0_MC - 1.96 * math.sqrt(s_var/M), V0_MC + 1.96 * math.sqrt(s_var/M)]
    return V0_BS, V0_MC, CI95

def call_payoff(S_T, K):
    return np.maximum(0, S_T-K)


EU_Option_BS_MC(110, 0.04, 0.2, 1, 100, 10000, call_payoff)

