"""
Navier–Stokes 1D Residual Minimization using RDSF

This script symbolically defines a trial solution for the 1D Navier–Stokes equation,
computes its residual, and performs a parameter sweep over ε and δ to minimize the
Mean Squared Error (MSE) of the residual using symbolic differentiation (via SymPy).

Author: Mohammad Nasiri
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sin, cos, diff, simplify, pi, lambdify
from tqdm import tqdm

# === Symbolic Definitions ===
x, t, epsilon, delta, nu = symbols('x t epsilon delta nu')

# Trial function
u = epsilon * sin(pi * x) * sin(pi * t) + delta * cos(pi * x) * cos(pi * t)

# Derivatives for 1D Navier–Stokes residual
u_t = diff(u, t)
u_x = diff(u, x)
u_xx = diff(u_x, x)

# Residual: u_t + u * u_x - ν * u_xx
residual = simplify(u_t + u * u_x - nu * u_xx)

# Lambdify the symbolic expression
residual_func = lambdify((x, t, epsilon, delta, nu), residual, 'numpy')

# === Parameter Sweep Settings ===
n_samples = 300
x_samples = np.random.uniform(0, 1, n_samples)
t_samples = np.random.uniform(0, 1, n_samples)

epsilon_range = np.linspace(0.001, 0.05, 30)
delta_range = np.linspace(0.001, 0.05, 30)
nu_val = 0.01  # viscosity

heatmap = np.zeros((len(epsilon_range), len(delta_range)))

# === Residual Sweep over (ε, δ) ===
print("Sweeping ε and δ for residual minimization...")

for i, eps in enumerate(tqdm(epsilon_range, desc="Sweeping ε")):
    for j, delt in enumerate(delta_range):
        res_vals = residual_func(x_samples, t_samples, eps, delt, nu_val)
        mse = np.mean(res_vals**2)
        heatmap[i, j] = mse

# === Best Parameters ===
min_idx = np.unravel_index(np.argmin(heatmap), heatmap.shape)
best_eps = epsilon_range[min_idx[0]]
best_delta = delta_range[min_idx[1]]
best_mse = heatmap[min_idx]

# === Plotting ===
plt.figure(figsize=(8, 6))
plt.imshow(heatmap.T,
           extent=[epsilon_range[0], epsilon_range[-1], delta_range[0], delta_range[-1]],
           origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='MSE of Residual')
plt.scatter(best_eps, best_delta, color='red', label='Best Parameters')
plt.xlabel(r'$\varepsilon$')
plt.ylabel(r'$\delta$')
plt.title('Residual MSE Heatmap – 1D Navier–Stokes (RDSF Trial)')
plt.legend()
plt.tight_layout()
plt.savefig("navier_stokes_rdsf_vs_fd_t100.jpg", dpi=300)
plt.show()

# === Print Result ===
print(f"Best ε: {best_eps:.6f}, Best δ: {best_delta:.6f}, Min Residual MSE: {best_mse:.2e}")
