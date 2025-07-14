"""
Allen–Cahn Equation: Residual Sweep over Epsilon and Delta (RDSF Framework)

This script explores the residual of a symbolic trial function:
    u(x, t) = ε·sin(x)·sin(t) + δ·cos(2x)·cos(t)
with respect to the Allen–Cahn PDE:
    u_t = ε² u_xx - (u³ - u)

It performs a grid search over ε and δ to find the parameter pair that minimizes
the Mean Squared Error (MSE) of the residual evaluated on randomly sampled (x, t) points.

Output: Residual heatmap and optimal parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==== General Settings ====
np.random.seed(42)
epsilon = 0.05  # Fixed PDE parameter (controls diffusion)

# Generate random sampling points (x, t)
n_samples = 300
x_vals = np.random.uniform(0, np.pi, n_samples)
t_vals = np.random.uniform(0, np.pi, n_samples)

# ==== Trial function and derivatives ====
def trial_solution(x, t, eps_param, delta_param):
    return eps_param * np.sin(x) * np.sin(t) + delta_param * np.cos(2 * x) * np.cos(t)

def du_dt(x, t, eps_param, delta_param):
    return eps_param * np.sin(x) * np.cos(t) - delta_param * np.cos(2 * x) * np.sin(t)

def d2u_dx2(x, t, eps_param, delta_param):
    return -eps_param * np.sin(x) * np.sin(t) - 4 * delta_param * np.cos(2 * x) * np.cos(t)

# ==== Residual of the PDE ====
def residual(x, t, eps_param, delta_param):
    u = trial_solution(x, t, eps_param, delta_param)
    dudt = du_dt(x, t, eps_param, delta_param)
    d2udx2 = d2u_dx2(x, t, eps_param, delta_param)
    return dudt - epsilon**2 * d2udx2 + (u**3 - u)

# ==== Sweep Grid ====
epsilons = np.linspace(0.001, 0.05, 30)
deltas = np.linspace(0.001, 0.05, 30)
residual_grid = np.zeros((len(epsilons), len(deltas)))

# ==== Compute Mean Residual for Each (ε, δ) Pair ====
for i, eps_param in enumerate(tqdm(epsilons, desc="Sweeping ε")):
    for j, delta_param in enumerate(deltas):
        res_vals = residual(x_vals, t_vals, eps_param, delta_param)
        mse = np.mean(res_vals**2)
        residual_grid[i, j] = mse

# ==== Find Optimal Parameters ====
min_idx = np.unravel_index(np.argmin(residual_grid), residual_grid.shape)
best_eps = epsilons[min_idx[0]]
best_delta = deltas[min_idx[1]]
best_mse = residual_grid[min_idx]

print(f"Best ε: {best_eps:.6f}, Best δ: {best_delta:.6f}, Min Residual MSE: {best_mse:.2e}")

# ==== Plot Residual Heatmap ====
plt.figure(figsize=(8, 6))
plt.imshow(residual_grid.T,
           extent=[epsilons[0], epsilons[-1], deltas[0], deltas[-1]],
           origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Mean Squared Residual')
plt.xlabel(r'$\varepsilon$')
plt.ylabel(r'$\delta$')
plt.title('Residual Heatmap for Allen–Cahn Equation (RDSF)')
plt.tight_layout()
plt.savefig('allen_cahn_rdsf_heatmap.jpg', dpi=300)
plt.show()
