"""
Navier–Stokes 1D Equation: Numerical Method Comparison

This script solves the 1D viscous Burgers-type form of the Navier–Stokes equation
using three approaches:
1. Finite Difference (FD) Method
2. Spectral Method using Discrete Sine Transform (DST)
3. Physics-Informed Neural Networks (PINN)

The script compares the final solution u(x, t=π) from each method and reports MSEs.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dst, idst
import torch
import torch.nn as nn

# ==== Problem Settings ====
nu = 0.01  # viscosity coefficient
Nx, Nt = 128, 1000
x_vals = np.linspace(0, np.pi, Nx)
t_vals = np.linspace(0, np.pi, Nt)
dx = x_vals[1] - x_vals[0]
dt = t_vals[1] - t_vals[0]

# ==== Theorem-Based Trial Function ====
def u_theorem(x, t, eps=0.001, delta=0.001):
    u = eps * np.sin(np.pi * x) * np.sin(np.pi * t) + delta * np.cos(np.pi * x) * np.cos(np.pi * t)
    return np.clip(u, 0, 1)

# ==== Finite Difference Method ====
u_fd = u_theorem(x_vals, 0)
for _ in range(Nt):
    u_x = (np.roll(u_fd, -1) - np.roll(u_fd, 1)) / (2 * dx)
    u_xx = (np.roll(u_fd, -1) - 2 * u_fd + np.roll(u_fd, 1)) / dx**2
    u_fd += dt * (-u_fd * u_x + nu * u_xx)
    u_fd[0] = u_fd[1]    # Neumann BC
    u_fd[-1] = u_fd[-2]
    u_fd = np.clip(u_fd, 0, 1)

# ==== Spectral Method ====
def spectral_solver(u0, dt, steps, nu):
    u = u0.copy()
    for _ in range(steps):
        u_hat = dst(u, type=2, norm='ortho')
        k = np.arange(1, len(u_hat) + 1)
        decay = np.exp(-nu * (k * np.pi / np.pi)**2 * dt)
        u_hat *= decay
        u = idst(u_hat, type=2, norm='ortho')
        u_x = np.gradient(u, dx)
        u -= dt * u * u_x
        u = np.clip(u, 0, 1)
    return u

u0 = u_theorem(x_vals, 0)
u_spectral = spectral_solver(u0, dt, Nt, nu)

# ==== PINN Definition ====
torch.manual_seed(0)
device = torch.device("cpu")

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==== Training Data ====
x_train = torch.FloatTensor(np.random.uniform(0, np.pi, 1000)).view(-1, 1) / np.pi
t_train = torch.FloatTensor(np.random.uniform(0, np.pi, 1000)).view(-1, 1) / np.pi
x_train.requires_grad_(True)
t_train.requires_grad_(True)

# ==== Loss Function ====
def pinn_loss(model, x, t):
    u = model(x, t)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return torch.mean((u_t + u * u_x - nu * u_xx)**2)

# ==== Train PINN ====
for epoch in range(1000):
    optimizer.zero_grad()
    loss = pinn_loss(model, x_train, t_train)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.2e}")

# ==== Evaluate PINN at t = π ====
x_eval = torch.FloatTensor(x_vals).view(-1, 1) / np.pi
t_eval = torch.FloatTensor([np.pi] * Nx).view(-1, 1) / np.pi
with torch.no_grad():
    u_pinn = model(x_eval, t_eval).view(-1).numpy()
    u_pinn = np.clip(u_pinn, 0, 1)

# ==== MSE Comparison ====
u_true = u_fd
u_theo = u_theorem(x_vals, np.pi)
mse_theo = np.mean((u_true - u_theo)**2)
mse_spectral = np.mean((u_true - u_spectral)**2)
mse_pinn = np.mean((u_true - u_pinn)**2)

print(f"MSE Theorem-Based: {mse_theo:.2e}")
print(f"MSE Spectral:       {mse_spectral:.2e}")
print(f"MSE PINN:           {mse_pinn:.2e}")

# ==== Plot Comparison ====
plt.figure(figsize=(8, 6))
plt.plot(x_vals, u_true, label='FD (Reference)', linewidth=2)
plt.plot(x_vals, u_theo, '--', label='Theorem-Based')
plt.plot(x_vals, u_spectral, '--', label='Spectral')
plt.plot(x_vals, u_pinn, '--', label='PINN')
plt.xlabel("x")
plt.ylabel("u(x, t=π)")
plt.title("Navier–Stokes: Method Comparison at t = π")
plt.legend()
plt.tight_layout()
plt.savefig("navier_stokes_comparison.png", dpi=300)
plt.show()
