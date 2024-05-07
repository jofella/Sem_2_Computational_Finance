# Exercise 01
# a) Write a python function CRR model
import math
import numpy as np
def CRR_stock(S_O,r,sigma,T,M):
    dt=T/M
    u=np.exp(sigma*np.sqrt(dt))
    d=1/u
    S=np.zeros((M+1,M+1))
    for i in range (M+1):
        for j in range (i+1):
            S[i,j]=S_O*u**j*d**(i-j)
    return S
#test parameters
s_O=100
r=0.05
sigma=0.2
T=1
M=4
stock = CRR_stock(S_O,r,sigma, T,M)
print(stock)
# b) CRR Model
# computes the price of european calls in the CRR model
def CRR_stock(S_0,r,sigma,T,M):
    #compute values of u, d and q
    dt=T/M
    alpha=np.exp(r*dt)
    beta = 1/2*(1/alpha+alpha*math.exp(math.pow(sigma,2)*dt))
    u=beta+math.sqrt(beta**2-1)
    d=1/u
    q=(alpha-d)/(u-d)
    
    # allocate matrix S
    S=np.empty((M+1,M+1))
    for i in range (M+1):
        for j in range (i+1):
            S[i,j]=S_0*u**j*d**(i-j)
    S[-1,:]=np.maximum(S[-1:]-K,0)/math.exp(M*r)
    return S[M,:]
    
    
#test parameters:
S_0=100
K=105
T=0.5
sigma=0.4
r=0.05
M=4
EuroCall= CRR_stock(S_0,r,sigma,T,M)
print(EuroCall)

#c) Compare the CRR model to the true price in the BS-model
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# BS-Model 
def Blackscholes_EuCall(t,S_t,r,sigma,T,K):
    d1=(np.log(S_t/K)+(r+sigma**2/2)*(T-t))/(sigma*np.sqrt(T-t))
    d2=d1-sigma*np.sqrt(T-t)
    phi_d1=norm.cdf(d1)
    phi_d2=norm.cdf(d2)
    V=S_t*phi_d1 - K*np.exp(-r*(T-t))*phi_d2
    return V
#test parameters:
t=0
S_t=100
r=0.05
sigma=0.4
T=0.5
K=105
V_BS=Blackscholes_EuCall(t,S_t,r,sigma,T,K)
print(V_BS)

# Exercise 2
# compute log_return
def log_returns(data):
    return np.diff(np.log(data))
# b)
# apply the function to the imported time series
dax= np.genfromtxt('time_series_dax_2024.csv', delimiter=';', usecols=4, skip_header=1)
dax= np.flip(dax)
dax_log_returns = log_returns(dax) #apply function to ts

#Visualize log returns
plt.clf()
plt.plot(dax_log_returns)

plt.title('Dax Log-returns (1990-2024)')
plt.xlabel('Trading days')
plt.ylabel('Daily Log-Returns')
plt.show()
