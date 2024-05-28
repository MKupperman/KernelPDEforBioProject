from typing import Union, Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def SIS_ODE(t, y, beta, gamma, f):
    I = y[0]
    dI = beta * I * (1 - I) - gamma * I + f(t)
    return [dI]

def SIS(dt, S0, I0, beta, gamma, T:Union[float, int], f:callable):
    N = S0 + I0  # Total population

    # Initial conditions
    I0 = I0 / N  # normalize from absolute to relative pop size
    y0 = [I0]

    # Time span
    t_span = (0, T)
    t_eval = np.arange(0, T, dt)

    # Solve the ODE with dense output
    sol = solve_ivp(SIS_ODE, t_span, y0, args=(beta, gamma, f), dense_output=True)

    # Evaluate the solution at specified time points
    I = sol.sol(t_eval)[0]
    t = t_eval

    return t, I

def main():
    # Parameters
    beta = 0.002  # Infection rate
    gamma = 0.001  # Recovery rate
    S0 = 990  # Initial susceptible population
    I0 = 10   # Initial infected population
    T = 60    # Total time to simulate
    dt = 0.1  # Time step

    # Run simulation
    t, I = SIS(dt, S0, I0, beta, gamma, T, lambda x: 0.0)
    S = (S0 + I0 * (S0 + I0)) * np.ones(len(t)) - I * (S0 + I0)  # conservation of population

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I * (S0 + I0), label='Infected')  # convert I back to absolute numbers
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIS Model Simulation')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
