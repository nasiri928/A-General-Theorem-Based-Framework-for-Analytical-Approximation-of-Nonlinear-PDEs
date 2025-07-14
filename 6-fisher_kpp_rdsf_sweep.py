"""
Residual Minimization for Fisher–KPP Equation Using RDSF

This script estimates the optimal parameters ε and δ for a trial function
by minimizing the residual of the Fisher–KPP equation:

    u_t = D·u_xx + r·u(1 - u)

It evaluates the squared residual over random points in the domain (x, t),
and visualizes the residual surface as a heatmap.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==== Equation Parameters ====
D = 1.0
r = 1.0
n_samples = 300
x_vals = np.pi * np.random.rand(n_samples)
t_vals = np.pi * np.random.rand(n_samples)

# ==== Sweep Ranges for ε and δ ====
epsilons = np.linspace(0.001, 0.05, 30)
deltas = np.linspace(0.001, 0.05, 30)
residual_grid = np.zeros((len(epsilons), len(deltas)))

# ==== Trial Function and Derivatives ====
def trial_u(x, t, eps, delta):
    return eps * np.sin(x) * np.sin(t) + delta * np.cos(2 * x) * np.cos(t)

def du_dt(x, t, eps, delta):
    return eps * np.sin(x) * np.cos(t) - delta * np.cos(2 * x) * np.sin(t)

def d2u_dx2(x, t, eps, delta):
    return -eps * np.sin(x) * np.sin(t) - 4 * delta * np.cos(2 * x) * np.cos(t)

# ==== Residual Minimization ====
print("Sweeping ε and δ...")
for i, eps in tqdm(enumerate(epsilons), total=len(epsilons), desc="Sweeping ε"):
    for j, delta in enumerate(deltas):
        residuals = []
        for x, t in zip(x_vals, t_vals):
            u_val = trial_u(x, t, eps, delta)
            res = du_dt(x, t, eps, delta) - D * d2u_dx2(x, t, eps, delta) - r * u_val * (1 - u_val)
            residuals.append(res ** 2)
        residual_grid[i, j] = np.mean(residuals)

# ==== Find Optimal Parameters ====
min_idx = np.unravel_index(np.argmin(residual_grid), residual_grid.shape)
best_eps = epsilons[min_idx[0]]
best_delta = deltas[min_idx[1]]
best_residual = residual_grid[min_idx]
print(f"Best ε: {best_eps:.6f}, Best δ: {best_delta:.6f}, Min Residual MSE: {best_residual:.2e}")

# ==== Plot Heatmap ====
plt.figure(figsize=(8, 6))
plt.imshow(residual_grid, extent=[deltas[0], deltas[-1], epsilons[-1], epsilons[0]],
           cmap='viridis', aspect='auto')
plt.colorbar(label='Mean Squared Residual')
plt.xlabel(r'$\delta$')
plt.ylabel(r'$\varepsilon$')
plt.title('RDSF Residual Heatmap for Fisher–KPP Equation')
plt.tight_layout()
plt.savefig("fisher_kpp_rdsf_heatmap.jpg", dpi=300)
plt.show()
