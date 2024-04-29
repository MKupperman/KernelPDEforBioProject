import pytest
from KBio.DataSimulators.SIS import SIS


def test_SIS():
    beta = 0.002  # Infection rate
    gamma = 0.001  # Recovery rate
    S0 = 990  # Initial susceptible population
    I0 = 10   # Initial infected population
    T = 60    # Total time to simulate
    dt = 0.1  # Time step

    I = SIS(dt, S0, I0, beta, gamma,T, f = lambda x: 1/1000)