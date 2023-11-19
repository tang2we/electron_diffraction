import numpy as np
import pandas as pd
from scipy.optimize import minimize

data = pd.read_excel('電子繞射.xlsx')
s1 = data['s1(cm)'].dropna()
d1 = 0.123
s2 = data['s2(cm)'].dropna()
d2 = 0.213
lambda_th = data['theoretical value (nm)'].dropna()
# print(s1)
# print(s2)
# print(lambda_th)

# tan_2theta = R*sin(s1/R)/(delta + R + R*cos(s1/R))
# solve equation
# tan_theta = -1+np.sqrt(1+tan_2theta**2)/tan_2theta
# sin_theta = tan_theta/np.sqrt(1+tan_theta**2)
# find the best solution of delta and R to make lambda_exp - lambda_th smallest

def wavelength_difference(params):
    R, delta = params
    tan_2theta1 = R * np.sin(s1 / R) / (delta + R + R * np.cos(s1 / R))
    tan_2theta2 = R * np.sin(s2 / R) / (delta + R + R * np.cos(s2 / R))
    tan_theta1 = (-1 + np.sqrt(1 + tan_2theta1**2)) / tan_2theta1
    tan_theta2 = (-1 + np.sqrt(1 + tan_2theta2**2)) / tan_2theta2
    sin_theta1 = tan_theta1 / np.sqrt(1 + tan_theta1**2)
    sin_theta2 = tan_theta2 / np.sqrt(1 + tan_theta2**2)
    lambda_exp1 = 2 * d1 * sin_theta1
    lambda_exp2 = 2 * d2 * sin_theta2
    lambda_exp = np.concatenate((lambda_exp1, lambda_exp2))
    lambda_th1 = np.concatenate((lambda_th, lambda_th))
    return np.sum(np.abs(lambda_exp - lambda_th1))

# Initial guess for R and delta
initial_guess = [6.5, 0.5]  # initial values for R and delta

# Perform optimization to minimize the difference
result = minimize(wavelength_difference, initial_guess, method='Nelder-Mead', bounds=((0, None), (0, None)))

# The optimized values for R and delta
best_R, best_delta = result.x

print(f"Best R: {best_R}, Best delta: {best_delta}")