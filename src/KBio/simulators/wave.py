import numpy as np
import matplotlib.pyplot as plt

def wave_eq_fdm(c, k, L, T, dx, dt):
    # Constants and setup
    x = np.arange(0, L+dx, dx)
    t = np.arange(0, T+dt, dt)
    r = c * dt / dx
    r_squared = r**2
    
    # Initialize the solution matrix
    u = np.zeros((len(t), len(x)))
    
    # Initial conditions
    u[0, :] = np.sin(np.pi * x / L)  # Initial displacement
    u[1, :] = u[0, :]  # Assuming initial velocity is zero
    
    # Finite difference solution
    for n in range(1, len(t) - 1):
        for j in range(1, len(x) - 1):
            u[n+1, j] = (2 * u[n, j] - u[n-1, j] +
                         r_squared * (u[n, j+1] - 2 * u[n, j] + u[n, j-1]) +
                         dt**2 * k * u[n, j]**2)  # Nonlinear term k*u^2

        # Boundary conditions
        u[n+1, 0] = 0  # Dirichlet BC
        u[n+1, -1] = 0  # Dirichlet BC

    return x, t, u

# Parameters
c = 1.0    # Wave speed
k = 0.01   # Nonlinear coefficient
L = 10     # Length of the domain
T = 10     # Total time
dx = 0.1   # Spatial step size
dt = 0.005 # Time step size (CFL condition: dt <= dx/c)

# Run the simulation
x, t, u = wave_eq_fdm(c, k, L, T, dx, dt)

# Plot results
plt.figure(figsize=(12, 8))
plt.imshow(u, aspect='auto', extent=[0, L, T, 0], cmap='viridis')
plt.colorbar(label='Wave amplitude')
plt.xlabel('Position (x)')
plt.ylabel('Time (t)')
plt.title('Nonlinear Wave Equation Simulation')
plt.show()