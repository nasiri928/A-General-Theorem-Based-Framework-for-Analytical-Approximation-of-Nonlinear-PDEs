"""
Comparison of Approximation Methods for the Fisher–KPP Equation

This script compares three approaches for approximating the solution to the 1D Fisher–KPP equation:
    u_t = D·u_xx + r·u(1 - u)

Methods compared:
- Theorem-based (symbolic RDSF trial function)
- Spectral method (Fourier sine basis)
- Physics-Informed Neural Network (PINN)

All methods are evaluated at t = π, and their results are compared with a finite difference reference solution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
import torch
import torch.nn as nn
from tqdm import tqdm

# ==== Parameters ====
D = 0.1
r = 1.0
Nx = 100
Nt = 1000
x = np.linspace(0, np.pi, Nx)
t = np.linspace(0, np.pi, Nt)
dx = x[1] - x[0]
dt = t[1] - t[0]
eps_star = 0.001
delta_star = 0.001
SAVE_DIR = "./"  # Output directory

# ==== Theorem-Based Trial Function ====
def theorem_solution(x, t, eps, delta):
    return np.clip(
        eps * np.sin(x[:, None]) * np.sin(t[None, :]) +
        delta * np.cos(2 * x[:, None]) * np.cos(t[None, :]),
        0, 1
    )

u_theorem = theorem_solution(x, t, eps_star, delta_star)

# ==== Reference: Finite Difference Solver ====
u_fd = np.zeros((Nx, Nt))
u_fd[:, 0] = u_theorem[:, 0]
for n in range(Nt - 1):
    u = u_fd[:, n]
    u_new = u.copy()
    u_new[1:-1] = u[1:-1] + dt * (
        D * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2 + r * u[1:-1] * (1 - u[1:-1])
    )
    u_new[0] = u_new[1]
    u_new[-1] = u_new[-2]
    u_fd[:, n + 1] = np.clip(u_new, 0, 1)

# ==== Spectral Method Solver ====
def spectral_solver(u0, D, r, dx, dt, Nt):
    N = len(u0)
    u = u0.copy()
    us = [u0.copy()]
    for _ in range(Nt - 1):
        u_hat = dst(u, type=2)
        k = np.arange(1, N + 1)
        u_xx = idst(-(k * np.pi / np.pi)**2 * u_hat, type=2) / (2 * N)
        u = u + dt * (D * u_xx + r * u * (1 - u))
        u = np.clip(u, 0, 1)
        us.append(u.copy())
    return np.array(us).T

u_spectral = spectral_solver(u_theorem[:, 0], D, r, dx, dt, Nt)

# ==== PINN ====
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x, t):
        return self.net(torch.cat((x, t), dim=1))

torch.manual_seed(0)
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

x_train = torch.FloatTensor(np.random.uniform(0, np.pi, (1000, 1)))
t_train = torch.FloatTensor(np.random.uniform(0, np.pi, (1000, 1)))
x_train.requires_grad = True
t_train.requires_grad = True

def pde_loss(model, x, t):
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    return loss_fn(u_t, D * u_xx + r * u * (1 - u))

# ==== Train PINN ====
for epoch in range(801):
    optimizer.zero_grad()
    loss = pde_loss(model, x_train, t_train)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.2e}")

# ==== Evaluate PINN at t = π ====
x_tensor = torch.FloatTensor(x).view(-1, 1)
t_pi = torch.FloatTensor(np.pi * np.ones_like(x)).view(-1, 1)
u_pinn = model(x_tensor, t_pi).detach().numpy().flatten()
u_pinn = np.clip(u_pinn, 0, 1)

# ==== Final Profiles ====
u_fd_final = u_fd[:, -1]
u_theo_final = u_theorem[:, -1]
u_spec_final = u_spectral[:, -1]

# ==== MSE Comparisons ====
mse_theo = np.mean((u_fd_final - u_theo_final)**2)
mse_spec = np.mean((u_fd_final - u_spec_final)**2)
mse_pinn = np.mean((u_fd_final - u_pinn)**2)

print(f"MSE Theorem-Based: {mse_theo:.2e}")
print(f"MSE Spectral:       {mse_spec:.2e}")
print(f"MSE PINN:           {mse_pinn:.2e}")

# ==== Plot ====
plt.figure(figsize=(8, 6))
plt.plot(x, u_fd_final, label='Finite Difference', linewidth=2)
plt.plot(x, u_theo_final, '--', label='Theorem-Based', linewidth=2)
plt.plot(x, u_spec_final, '-.', label='Spectral Method', linewidth=2)
plt.plot(x, u_pinn, ':', label='PINN', linewidth=2)
plt.xlabel("x")
plt.ylabel("u(x, π)")
plt.title("Fisher–KPP Equation: Comparison at t = π")
plt.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR + "fisher_kpp_compare_methods.png", dpi=300)
plt.close()
