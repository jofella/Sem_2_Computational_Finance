# C-Exercise 23, SS 2024
# Group: QF2
# Maris Leibold, Josef Fella

import math
import cmath
import scipy.integrate


#Script p. 52
def chi_t(u, S0, r, T, sigma):
    return cmath.exp(complex(0,u)*(math.log(S0+r*T)-(complex(0,u)+u**2)*(sigma**2/2)*T))

#Script euqation 4.6
def lap_trans(z, K):
    return K**(1-z)/(z*(z-1))


def BS_EuCall_Laplace(S0, r, sigma, T, K, R):
    int_part = scipy.integrate.quad(lap_trans(R+complex(0,u)*chi_t()))


u=2
z=complex(0,u)