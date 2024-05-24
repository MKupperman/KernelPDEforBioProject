from typing import Union, Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def _SIS_ODE(t, y, beta, gamma, f):
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
    # sol = solve_ivp(_SIS_ODE, t_span, y0, args=(beta, gamma, f), t_eval=t_eval, method='RK45')
    sol = solve_ivp(_SIS_ODE, t_span, y0, args=(beta, gamma, f), dense_output=True, method='RK45')

    # Evaluate the solution at specified time points
    # I = sol.y[0]
    I = sol.sol(t_eval)[0]
    t = t_eval
    return t, I


def SIS_old(dt, S0, I0, beta, gamma, T:Union[float, int], f:callable):
    N = S0 + I0  # Total population
    t = np.arange(0, T, dt)
    I = np.zeros(len(t))
    I[0] = I0 / N  # normalize from absolute to relative pop size

    for i in range(1,len(t)):
        dI = (beta * I[i-1] * (1 - I[i-1]) - gamma * I[i-1]) + f(t[i-1])
        if dI + I[i-1] > 1:
            print("Step size: ", dt)
            print("Current time: ", t[i-1])
            print("Current infected: ", I[i-1])
            print("Current dI: ", dI)
            raise Exception("Step size too large! Simulation unstable. Reduce dt.")
        I[i] = I[i-1] + dt * dI
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
    S = (S0 + I0) * np.ones(len(t)) - I  # conservation of population

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIS Model Simulation')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()