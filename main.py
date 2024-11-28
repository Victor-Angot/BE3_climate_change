import numpy as np
import matplotlib.pyplot as plt
from math import *
import random
import pandas as pd

# Constants
d1, d2, d3 = 0.323, 0.015, 0.5
Q0 = 343  # Baseline solar radiation (W/m²)
beta = 8e-4
T_ecc = 1e5  # Eccentricity period (years)
omega_ecc = 2 * np.pi / T_ecc  # Eccentricity cycle frequency (~100,000 years)
phi_ecc = np.pi * 290/180  # Phase offset
lamb = 7e-4
T_obl = 41000
omega_obl = 2*np.pi/T_obl
phi_obl = 250*np.pi/180
A, B = 210, 1.85  # Constants in the temperature equation
c = 3e8  # Heat capacity

epsilon_n = 0.021  # Noise strength

# Albedo function
def albedo(T):
    T_0 = 12
    return d1 - d2 * np.tanh(d3 * (T - T_0))

# Solar forcing function
def solar_forcing(t, with_obliquity=False):
    f_ecc = beta * np.cos(omega_ecc * t + phi_ecc)
    if with_obliquity:
         f_obl = lamb * np.cos(omega_obl * t + phi_obl)
    else:
        f_obl = 0  
    return Q0 * (1 + f_ecc + f_obl)

# Stochastic differential equation solver
def equation_solver(T0, dt=10, with_obliquity=False, with_stochastic=False, with_superposition=False):
    # Time discretization
    time = np.arange(0, 800000, dt)  
    if with_superposition:
        path = 'data_delta18O.tab'
        data = pd.read_csv(path, delimiter='\t')
        time = np.array(data.loc[:,'Age [ka BP]'].tolist())*1000
    n_steps = len(time)
    # Initial condition
    T = T0  # Initial temperature (°C)
    temperatures = [T]
    # Solve the equation
    var = dt 
    dt = dt * 3600 * 24 * 365.25  # Convert years to seconds
    for t in time[:-1]:
        Q = solar_forcing(t, with_obliquity)
        alpha = albedo(T)
        deterministic = (Q * (1 - alpha) - A - B * T) * dt / c 
        if not with_stochastic:
            stochastic = 0
        else:
            stochastic = random.gauss(0,1) * np.sqrt(var) * np.sqrt(epsilon_n)
        dT = deterministic + stochastic
        T += dT
        temperatures.append(T)
    return temperatures, time

scenario = 6

if scenario == 1:
    T0 = 12.0  # Reference temperature (°C)
    temperatures, time = equation_solver(T0, with_obliquity=False, with_stochastic=False)
    plt.figure(figsize=(10, 6))
    plt.plot(time, temperatures, label='Temperature (°C)')
    plt.xlabel('Time (years)')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Evolution with Deterministic Climate Model')
    plt.grid()
    plt.legend()
    plt.show()

elif scenario == 2:
    T0 = 14.0  # Reference temperature (°C)
    temperatures_warm1, time = equation_solver(T0, with_obliquity=False, with_stochastic=False)
    T0 = 15.0  # Reference temperature (°C)
    temperatures_warm2, time = equation_solver(T0, with_obliquity=False, with_stochastic=False)
    T0 = 10.0  # Reference temperature (°C)
    temperatures_cold1, time = equation_solver(T0, with_obliquity=False, with_stochastic=False)
    T0 = 11.0  # Reference temperature (°C)
    temperatures_cold2, time = equation_solver(T0, with_obliquity=False, with_stochastic=False)
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(time, temperatures_warm1, label='Warm mode (T0 = 14°C)')
    plt.plot(time, temperatures_warm2, label='Warm mode (T0 = 15°C)')
    plt.plot(time, temperatures_cold1, label='Cold mode (T0 = 10°C)')
    plt.plot(time, temperatures_cold2, label='Cold mode (T0 = 11°C)')
    plt.xlabel('Time (years)')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Evolution with Deterministic Climate Model')
    plt.grid()
    plt.legend()
    plt.show()

elif scenario == 3:
    T0 = 14.0  # Reference temperature (°C)
    temperatures_warm1, time = equation_solver(T0, with_obliquity=True, with_stochastic=False)
    T0 = 15.0  # Reference temperature (°C)
    temperatures_warm2, time = equation_solver(T0, with_obliquity=True, with_stochastic=False)
    T0 = 10.0  # Reference temperature (°C)
    temperatures_cold1, time = equation_solver(T0, with_obliquity=True, with_stochastic=False)
    T0 = 11.0  # Reference temperature (°C)
    temperatures_cold2, time = equation_solver(T0, with_obliquity=True, with_stochastic=False)
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(time, temperatures_warm1, label='Warm mode (T0 = 14°C)')
    plt.plot(time, temperatures_warm2, label='Warm mode (T0 = 15°C)')
    plt.plot(time, temperatures_cold1, label='Cold mode (T0 = 10°C)')
    plt.plot(time, temperatures_cold2, label='Cold mode (T0 = 11°C)')
    plt.xlabel('Time (years)')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Evolution with Deterministic Climate Model and obliquity')
    plt.grid()
    plt.legend()
    plt.show()

elif scenario == 4:
    T0 = 14.0  # Reference temperature (°C)
    temperatures, time = equation_solver(T0, with_obliquity=True, with_stochastic=True)
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(time, temperatures, label='T0 = 14°C')
    plt.xlabel('Time (years)')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Evolution with Stochastic Climate Model')
    plt.grid()
    plt.legend()
    plt.show()

elif scenario == 5:
    T0 = 14.0  # Reference temperature (°C)
    epsilon_n_list = [0.01, 0.015, 0.03]
    T_ecc_list = [5e4, 1e5, 2e5]
    epsilon_n = epsilon_n_list[0]
    temperatures_eps0, time = equation_solver(T0, with_obliquity=True, with_stochastic=True)
    epsilon_n = epsilon_n_list[1]
    temperatures_eps1, time = equation_solver(T0, with_obliquity=True, with_stochastic=True)
    epsilon_n = epsilon_n_list[2]
    temperatures_eps2, time = equation_solver(T0, with_obliquity=True, with_stochastic=True)
    epsilon_n = 0.021
    T_ecc = T_ecc_list[0]
    temperatures_ecc0, time = equation_solver(T0, with_obliquity=True, with_stochastic=True)
    T_ecc = T_ecc_list[1]
    temperatures_ecc1, time = equation_solver(T0, with_obliquity=True, with_stochastic=True)
    T_ecc = T_ecc_list[2]
    temperatures_ecc2, time = equation_solver(T0, with_obliquity=True, with_stochastic=True)
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(time, temperatures_eps0, label='$\epsilon_n$ = 0.01')
    plt.xlabel('Time (years)')
    plt.ylabel('Temperature (°C)')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(time, temperatures_eps1, label='$\epsilon_n$ = 0.015')
    plt.xlabel('Time (years)')
    plt.ylabel('Temperature (°C)')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(time, temperatures_eps2, label='$\epsilon_n$ = 0.03')
    plt.xlabel('Time (years)')
    plt.ylabel('Temperature (°C)')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(time, temperatures_ecc0, label='$T_{ecc}$ = 50,000 years')
    plt.xlabel('Time (years)')
    plt.ylabel('Temperature (°C)')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(time, temperatures_ecc1, label='$T_{ecc}$ = 100,000 years')
    plt.xlabel('Time (years)')
    plt.ylabel('Temperature (°C)')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(time, temperatures_ecc2, label='$T_{ecc}$ = 200,000 years')
    plt.xlabel('Time (years)')
    plt.ylabel('Temperature (°C)')
    plt.grid()
    plt.legend()
    plt.show()

elif scenario == 6:
    path = 'data_delta18O.tab'
    data = pd.read_csv(path, delimiter='\t')
    T0 = 14.0  # Reference temperature (°C)
    temperatures, time = equation_solver(T0, dt=1,  with_obliquity=True, with_stochastic=True, with_superposition=True)
    time = np.array(data.loc[:,'Age [ka BP]'].tolist())*1000
    temperatures = np.array(temperatures)
    mean = np.mean(temperatures)
    exp = np.array(data.loc[:,'δ18O [‰]'].tolist())
    ratio = (max(temperatures)-min(temperatures))/(max(exp)-min(exp))
    exp_scale = exp + mean
    mean_exp_scale = np.mean(exp_scale)
    exp_scale = (exp_scale - mean_exp_scale) * ratio + mean_exp_scale
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(time, temperatures, label='T0 = 14°C')
    plt.plot(time, exp_scale, label='Experimental data')
    plt.xlabel('Age (kyears)')
    plt.legend()
    plt.legend(loc = 'lower right')
    plt.show()
    data = pd.read_csv(path, delimiter='\t')