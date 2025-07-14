"""
Residual Sweep for Burgers' Equation Using RDSF Trial Functions

This script evaluates the squared residual of a trial solution 
for the 1D Burgers' equation:
    u_t + u·u_x = D·u_xx

It performs a parameter sweep over trial function coefficients ε and δ, 
computes the residual at random (x,t) samples, and visualizes 
the MSE residual as a heatmap.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==== PDE Parameters ====
D = 0.1  # Viscosity term
n_samples = 300
epsilon_values = np.linspace(0.001, 0.05, 30)
delta_values = np.linspace(0.001, 0.05, 30)
x_range = t_range = (0, np.pi)

# ==== Trial Function ====
def trial_function(x, t, epsilon, delta):
    return epsilon * np.sin(x) * np.sin(t) + delta * np.cos(2 * x) * np.cos(t)

# ==== Residual of Burgers' Equation ====
def burgers_residual(x, t, u):
    du_dt = np.cos(x) * np.cos(t) * epsilon - np.sin(2 * x) * np.sin(t) * delta
    du_dx = np.cos(x) * np.sin(t) * epsilon - 2 * np.sin(2 * x) * np.cos(t) * delta
    d2u_dx2 = -np.sin(x) * np.sin(t) * epsilon - 4 * np.cos(2 * x) * np.cos(t) * delta
    return du_dt + u * du_dx - D * d2u_dx2

# ==== Residual Sweep ====
heatmap = np.zeros((len(epsilon_values), len(delta_values)))

print("Sweeping ε and δ over grid...")
for i, epsilon in enumerate(tqdm(epsilon_values)):
    for j, delta in enumerate(delta_values):
        residuals = []
        for _ in range(n_samples):
            x = np.random.uniform(*x_range)
            t = np.random.uniform(*t_range)
            u = trial_function(x, t, epsilon, delta)
            res = burgers_residual(x, t, u)
            residuals.append(res**2)
        mse = np.mean(residuals)
        heatmap[i, j] = mse

# ==== Optimal Parameters ====
min_idx = np.unravel_index(np.argmin(heatmap), heatmap.shape)
best_epsilon = epsilon_values[min_idx[0]]
best_delta = delta_values[min_idx[1]]
min_mse = heatmap[min_idx]
print(f"Best ε: {best_epsilon:.6f}, Best δ: {best_delta:.6f}, Min Residual MSE: {min_mse:.2e}")

# ==== Plot Heatmap ====
plt.figure(figsize=(8, 6))
sns.heatmap(
    np.log10(heatmap),
    xticklabels=np.round(delta_values, 3),
    yticklabels=np.round(epsilon_values, 3),
    cmap='viridis'
)
plt.title("Log10 MSE Residual Heatmap - Burgers' Equation (RDSF)")
plt.xlabel("δ")
plt.ylabel("ε")
plt.tight_layout()
plt.savefig("burgers_rdsf_heatmap.jpg", dpi=300)
plt.show()
