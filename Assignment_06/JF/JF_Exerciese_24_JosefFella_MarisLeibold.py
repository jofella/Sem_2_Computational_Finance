# C-Exercise 09, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella

import math
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from scipy import integrate



# first ode steps (Understanding how the solver in python works) // ref: https://www.youtube.com/watch?v=1fOqlbmWlPc - one of 10 vids...
def f(t, r): # beware of argument order
    x, y = r
    fx = np.cos(y)
    fy = np.sin(x)
    return fx, fy

sol = integrate.solve_ivp(f, (0, 10), (1, 1))
x, y = sol.y
print(x,y) # output arrays




# Pricing of Option

def BS_AmPerpPut_ODE(S_max, N, r, sigma_square, K):
    
    # 1.Step: Set up framework 
    
    # Compute x* - some sort of "hurdle" to reach
    x_star = (2*K*r) / (2*r+sigma_square)
    
    # Set equidistant grid - range of stock prices
    stock_grid = np.arange(0, S_max+1, 1) #t_span
    
    # 2.Step: ODE function - essentially the thing to be solved 
    # my understanding: We can express v(x) = v[0], and v'(x) = v[1] etc. 
    # --> AIM: bring it in a form that python understands ?!
    
    def v(x, v): # t=time, x=current state option price
        v_0 = v[0] # value we want to get
        v_1 = v[1]
        v_2 = v[2]
        return [v_1, (sigma_square / 2) * x**2 * v_2 + r*v_1 - r*v_0] #essentially we want to output an array of our solutions
    
    
    # 3.Step: Put option function - I guess thats the inital condition...
    def g(x):
        return np.maximum(k-x, 0)
    
    
    # Final solution
    result = integrate.solve_ivp(fun=v[0], t_span=[stock_grid[0], stock_grid[-1]], y0=g(x))
    # v[0] = v(x) function we want solution for, s_span = range to get values from, 
    # y0 = initial condition for x* >= x simple option payoff, else dynamics by ODE
    
    
    return result #like this it will output more than one array



# Visuals

# Set parameters
S_max = 200
N = 200
r = 0.05
k = 100
sigma_square = 0.4
K = 100


# Set up data
stock_grid = np.arange(0, S_max+1, 1) #could also be included as output of BS_AmPerpPut_ODE: optionprices, stock_grid = BS(...)
option_prices = BS_AmPerpPut_ODE(S_max, N, r, sigma_square, K)


# Plot data
plt.clf()
plt.plot(stock_grid, option_prices)

plt.title('Perpetual AM option on different stock prices')
plt.xlabel('asset price')
plt.ylabel('option prices')

plt.show()


# well no output due to wrong solver application