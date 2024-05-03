from typing import Union, Callable

import numpy as np
import matplotlib.pyplot as plt


def SIS(dt, S0, I0, beta, gamma, T:Union[float, int], f:callable):
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
        I[i] = I[i-1] + dI
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