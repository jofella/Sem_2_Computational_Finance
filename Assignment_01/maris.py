#Exercise 1
#a
import math
import numpy as np
import matplotlib.pyplot as plt
def CRR_stock(S_0, r, sigma, T, M):
    matrix = np.empty((M+1, M+1), dtype=float) #set up price matrix in appropriate shape
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


S = CRR_stock(100, 0.03, 0.3, 1, 100)

#1b

def CRR_EuCall(S_0, r, sigma, T, M, K):

    delta_t = T / M
    # Skript: 1.4 to 1.7 - Setting u, d, q
    beta = 0.5 * (math.exp(-r * delta_t) + math.exp((r + np.power(sigma, 2)) * delta_t))
    u = beta + np.sqrt(np.power(beta, 2) - 1)
    d = np.power(u, -1)
    q = (math.exp(r * delta_t) - d) / (u - d)

    #Gwet stock prices with previous function
    S = CRR_stock(S_0, r, sigma, T, M)

    #calculate call value at maturity
    Call_Value = np.array((1, M+1))
    Call_Value[:, M] = np.maximum(0, S[:,M]-K)

    #now loop over all previous steps and calculate value for that step (According to 1.14)

    for i in range(M-1, 0, -1):
        for j in range(i):
            Call_Value[:,i] = np.maximum(np.maximum(S[j,i]-K), np.exp(-r*delta_t)(q*()))


    return Call_Value


V = CRR_EuCall(100, 0.03, 0.3, 1, 100, 110)





###Exercise 2
#a
def log_returns(data):
    return np.diff(np.log(data))

#b
# 1.Step: Import data
dax = np.genfromtxt('Assignement_01/time_series_dax_2024.csv',
                    delimiter = ';'
                    , usecols = 4,
                    skip_header = 1)


# 2.Step: Flip ts
dax = np.flip(dax)

# 3.Step: Apply function to ts
dax_log_returns = log_returns(dax)

# 4.Step: Visualize log returns

plt.clf()
plt.plot(dax_log_returns)

plt.title('DAX Log-Returns (1990-2024)')
plt.xlabel('Trading days')
plt.ylabel('Daily Log-Return')

plt.show()

# 5.Step: calculate empirical mean and sd
def emp_mean(data, trading_days):
    N = len(data)
    sum_lk = sum(data)-data[0]
    result = trading_days/(N-1)*sum_lk
    return result

def emp_sd(data, trading_days):
    mean = emp_mean(data, trading_days)
    sum_lk = 0
    N = len(data)
    for k in range(1,N):
        sum_lk += np.power(data[k]-(mean/trading_days), 2)
    sd = np.power((trading_days/(N-2))*sum_lk, 0.5)
    return sd

dax_emp_mean = emp_mean(dax_log_returns, 250)
dax_emp_sd = emp_sd(dax_log_returns, 250)

print("Empirical mean: " + str(dax_emp_mean))
print("Empirical standard deviation: " + str(dax_emp_sd))

#c simulate normally distributed log return
dax_sim_log_returns = np.random.normal(dax_emp_mean, dax_emp_sd, len(dax_log_returns))
plt.plot(dax_sim_log_returns, c = "red")

#plot
plt.clf()
plt.plot(dax_sim_log_returns, color='blue')
plt.plot(dax_log_returns, color='red')

plt.title('Comparison Simulated versus observed log returns')
plt.xlabel('Trading Days')
plt.ylabel('Return')

plt.show()



#d
#The dax log returns are vay more volatile then the normally distributed one with an upward bias.
#Thus the normal distribution might not be a good approximation
#The real distribution seems to posses fatter tales, i.e. leptokurtosis
