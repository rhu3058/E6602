import numpy as np
import scipy.linalg as la

# Define the system parameters
age_structure=[0.1202,0.5558,0.324]
population = 51628117
population_growth = 1.3
initial_population = [i*population for i in age_structure]
m = 3  # number of age groups
T = 50  # number of years to simulate
d = np.array([0.08, 0.05, 0.1])  # death rates for each age group
b = np.array([0.45, 0, 0])  # female fertility rate pattern among different age groups.
A = np.zeros((m, m))
B = np.zeros((m, m))
B[:, 1] = b
A = np.array([[0,0,0],[0.975,0,0],[0,0.85,0]])
print(A)
print(B)
# Define the cost function
def cost(beta):
    x = np.zeros((m, T+1))
    x[:, 0] = np.array(initial_population)  # initial population for each age group
    for t in range(T):
        x[:, t+1] = A.dot(x[:, t]) + beta*B.dot(x[:, t])
    return np.sum((x[:, T] - np.array([i*population_growth for i in initial_population]))**2)

# Solve for the optimal beta using a nonlinear optimization algorithm
from scipy.optimize import minimize_scalar
result = minimize_scalar(cost, bounds=(0, 3), method='bounded')
beta_opt = result.x

# Simulate the system using the optimal beta
x = np.zeros((m, T+1))
x[:, 0] = np.array([1000000, 500000, 250000])
for t in range(T):
    x[:, t+1] = A.dot(x[:, t]) + beta_opt*B.dot(x[:, t])
print(beta_opt)
# Plot the population growth for each age group
import matplotlib.pyplot as plt
plt.plot(range(T+1), x[0, :], label='Age group 0-14')
plt.plot(range(T+1), x[1, :], label='Age group 14-55')
plt.plot(range(T+1), x[2, :], label='Age group 55 + ')
plt.legend()
plt.xlabel('Time (years)')
plt.ylabel('Population')
plt.show()
