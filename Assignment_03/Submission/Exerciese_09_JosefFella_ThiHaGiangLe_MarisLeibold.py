# C-Exercise 09, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella, Thi Ha Giang Le


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def Sample_TruncNormal_AR(a, b, mu, sigma, N):
    div = (sigma*(stats.norm.cdf((b-mu)/sigma)-stats.norm.cdf((a-mu)/sigma))) #as this stays constant just calculate once at beginning and save in a variable
    sample = np.zeros(N, dtype=float)

    #estimate c based on f(x)=<c*g(x) <=> c <= f(x)/g(x)

    def f(x, mu, sigma):
        return stats.norm.pdf((x - mu) / sigma) / div
    def g(x):
        return stats.uniform.pdf(x, a, b-a)
    x_range = np.linspace(a, b, 100)
    ratio = f(x_range, mu, sigma)/g(x_range)
    c = np.max(ratio)


    i = 0

    while i < N:
        #sample Y and U independently, both from Uniform distribution with support [a,b] and [0,1]
        y = np.random.uniform(a, b, 1)
        u = np.random.uniform(0, 1, 1)

        #generate the probabilities for the "target" distribution and the distribution of Y (Uniform in this example) for value y
        f_y = stats.norm.pdf((y - mu) / sigma) / div
        g_y = stats.uniform.pdf(y, a, b-a)

        #calculate ration of "target" distribution and the distribution of Y*c and check if smaller than u
        s = f_y / (c * g_y)

        if u < s:
            sample[i] = y
            i += 1
            print(i) #print i to see progress in real time, had some issues with my laptop
    return sample

#variables
a = 0
b = 2
mu = 0.5
sigma = 1
N = 10000

sample_AR = Sample_TruncNormal_AR(a, b, mu, sigma, N)


# Visualize data
plt.clf()
plt.hist(sample_AR, density=True)
x_range = np.linspace(-1,3,1000)
plt.plot(x_range, stats.truncnorm.pdf(x_range, (a-mu)/sigma, (b-mu)/sigma, mu, sigma), "r")
plt.show()