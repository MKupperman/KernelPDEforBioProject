import numpy as np
import matplotlib.pyplot as plt

def sis_model(dt, S0, I0, beta, gamma, T):
    N = S0 + I0  # Total population
    t = np.arange(0, T, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    
    S[0] = S0
    I[0] = I0
    
    # Simulate the dynamics
    for i in range(1, len(t)):
        dS = (-beta * S[i-1] * I[i-1] + gamma * I[i-1]) * dt
        dI = (beta * S[i-1] * I[i-1] - gamma * I[i-1]) * dt
        S[i] = S[i-1] + dS
        I[i] = I[i-1] + dI
    
    return t, S, I

# Parameters
beta = 0.002  # Infection rate
gamma = 0.001  # Recovery rate
S0 = 990  # Initial susceptible population
I0 = 10   # Initial infected population
T = 60    # Total time to simulate
dt = 0.1  # Time step

# Run simulation
t, S, I = sis_model(dt, S0, I0, beta, gamma, T)

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