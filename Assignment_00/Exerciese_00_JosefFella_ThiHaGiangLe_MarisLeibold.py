#QF Group 02
#Josef Fella
#Thi Ha Giang Le
#Maris Leibold


# Python Introduction - First steps
# This is a script that will help you to get to know the basic elements of Python
# Please replace all ??? within this file with your solutions

# Please make sure you followed the instructions on how to install Python and please make sure the following packages are correctly installed: Numpy, Scipy, Matplotlib


# 1. Importing packages
# using the import-statement to get access to functions from different packages
# different ways to import packages
# import the whole package
import math
# import the whole package with a new name
import numpy as np
import matplotlib.pyplot as plt

# 2. Real variables and basic mathematical operations
# set x to 5 and y to 3 and fill in the correct operations
x = 5
y = 3
print('The sum of x and y is: ' + str(x + y))
print('The difference of x and y is: ' + str(x - y))
print('The product of x and y is: ' + str(x * y))
print('The quotient of x and y is: ' + str(x / y))

# Hint: for the following you can use the power-function from the math package
print('x to the power of 1.5 is: ' + str(math.pow(x, 1.5)))

# 3. Array initialization
# In this section you will learn how to initialize Arrays and basic calculations on it.
# Initialize X as an Array with place for ten floating point numbers using the empty-function from the numpy package
X = np.empty(10, dtype = float)
# now save in X the vector [9,8,7,6,5,4,3,2,1,0]
X[:] = [9,8,7,6,5,4,3,2,1,0]
# initialize Y as the following Array [0 1 2 3 4 5 6 7 8 9]
Y = np.arange(10)
# transpose X
X.transpose()
# calculate the scalarproduct of X and Y
print('The scalarproduct of X and Y is: ' + str(np.dot(X, Y)))
# initialize Z as array of ones with size (1,5)
Z = np.ones([1, 5])
# multiply Z by 3
print(np.dot(Z, 3))
# initialize V as array of zeros with size (5,1)
V = np.zeros((5, 1))
print(V)
# initialize R as the range from 8 to 24 (including 24) with a stepsize of 2
R = np.arange(8, 25, 2)
# print the fourth entry of R
print('The fourth entry of R is: ' + str(R[3]))
# initialize L as the linespace from -2.5 to 2.5 with 50 samples
L = np.linspace(-2.5, 2.5, 50)
# print the 20th entry of L
print('The 20th entry of L is: ' + str(L[19]))

# 4. Matrices
# define the matrix consisting of 2*X, X+Y and Y^3 (use the numpy power-function)
M = np.matrix([X * 2, X + Y, np.power(Y, 3)])
print(M)
# create a matrix containing ones with size (10,3)
O = np.matrix(np.ones((10, 3)))
# multiply M with O, use either the dot-function or the matmul-function from the numpy package
print('The matrixproduct of M and O is: ' + str(np.dot(M, O)))
# multiply M with Y
print('The product of M and Y is: ' + str(np.dot(M, Y)))
# create a 5x5 matrix filled with N(0,1)-distributed values
# Hint: Check out the random-functions of numpy
G = np.matrix(np.random.normal(0, 1, (5, 5)))
print(G)

# accessing array-elements
# complete the following statements
print('The 5th element in the 5th row of G is: ' + str(G[4, 4]))
print('The first row of G is: ' + str(G[0, :]))
print('The third column of G is: ' + str(G[:, 2]))

# 5. Branching
# True-False Statements
# Give a True statement for each operator using x and y: <, >=, ==, !=
print(y < x)
print(x >= y)
print(x == x and y == y)
print(x != y)

# Write an if-else statement which compares z to x and y and prints one of the following messages
z = 4
if z == x:
    print('z is equal to x')
elif z == y:
    print('z is equal to y')
else:
    print('z is not equal to x nor y')

# 6. For-loop
# Use a for-loop to calculate the sum of the first 50 even numbers (starting with 0)
sum = 0
for i in range(0, 100, 2):
    sum += i

print('The sum of the first 50 even numbers is: ' + str(sum))


# 7. Custom functions
# define a function polynom which calculates the result of p^2+3p+2
def polynom(p):
    result = p**2 + 3*p + 2
    return result

print(polynom(5))


# implement the function h(x,y) = (x/y, exp(x))
def h(x, y):
    result = (x / y, math.e**x)
    return result

print(h(1, 1))

# 8. Importing data
# load DAX data using the genfromtxt-function from numpy
# we're only interested in the value and not the date, therefore only import the fifth column and skip the header
dax = np.genfromtxt(r"C:\Users\leibo\OneDrive - stud.uni-goettingen.de\Uni\Master\SS24\Compfin\Homework Sheets\time_series_dax_2024.csv", delimiter = ';', usecols = 4, skip_header = 1) #insert appropriate file path

# 9. Flip the data
# since the timeseries is antichronological you need to flip it using the flip-function from numpy.
dax = np.flip(dax)

# 10. Plotting
# plot the DAX data and label the axis

plt.plot(dax)
plt.xlabel("Days")
plt.ylabel("DAX value")
plt.title("DAX Closing Prices from 1990-2020")
plt.show()

# C-Exercise 00
def bond_value(V0, r, n, M, c):
    if r > 0 and V0 > 0:
        if c == 1:
            result = V0 * math.exp(r*n)
            return result
        elif c == 0:
            result = V0 * ( 1 + r/M) ** (n*M)
            return result
        else:
            return "c must be 0 or 1"
    else:
        return "Check inputs"

print(bond_value(1000, 0.05, 10, 4, 0))