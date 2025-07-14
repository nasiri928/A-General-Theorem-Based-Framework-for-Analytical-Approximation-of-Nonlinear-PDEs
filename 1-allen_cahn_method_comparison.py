"""
Allen–Cahn Equation: Comparison of Theorem-Based Approximation, Spectral Method, and Physics-Informed Neural Networks (PINNs)

This script solves the 1D Allen–Cahn equation:
    u_t = ε² u_xx - (u³ - u)
over the domain x ∈ [0, π], t ∈ [0, π].

Three methods are compared:
1. A symbolic trial function based on the theoretical result (FADF framework).
2. A spectral method using Discrete Sine Transform (DST).
3. A Physics-Informed Neural Network (PINN) approach.

The Mean Squared Error (MSE) of each method is evaluated at t = π
using a finite-difference solution as the reference.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dst, idst
import torch
import torch.nn as nn

# ==== Problem Parameters ====
epsilon = 0.01
domain = (0, np.pi)
T = np.pi
Nx, Nt = 128, 1000
x_vals = np.linspace(*domain, Nx)
t_vals = np.linspace(0, T, Nt)
dx = x_vals[1] - x_vals[0]
dt = t_vals[1] - t_vals[0]

# ==== Theorem-Based Trial Function ====
def u_theorem(x, t, eps=0.001, delta=0.001):
    u = eps * np.sin(x) * np.sin(t) + delta * np.cos(2*x) * np.cos(t)
    return np.clip(u, 0, 1)

# ==== Reference Finite-Difference Solver ====
u_fd = u_theorem(x_vals, 0)
for _ in range(Nt):
    u_xx = (np.roll(u_fd, -1) - 2*u_fd + np.roll(u_fd, 1)) / dx**2
    u_fd = u_fd + dt * (epsilon**2 * u_xx - (u_fd**3 - u_fd))
    u_fd[0] = u_fd[1]
    u_fd[-1] = u_fd[-2]
    u_fd = np.clip(u_fd, 0, 1)

# ==== Spectral Method ====
def spectral_allen_cahn(u0, T, dt, dx, epsilon):
    u = u0.copy()
    Nt = int(T / dt)
    for _ in range(Nt):
        u_hat = dst(u, type=2, norm='ortho')
        k = np.arange(1, len(u_hat)+1)
        decay = np.exp(-epsilon**2 * (k * np.pi / (domain[1] - domain[0]))**2 * dt)
        u_hat *= decay
        u = idst(u_hat, type=2, norm='ortho')
        u -= dt * (u**3 - u)
        u = np.clip(u, 0, 1)
    return u

u0 = u_theorem(x_vals, 0)
u_spectral = spectral_allen_cahn(u0, T, dt, dx, epsilon)

# ==== PINN Setup ====
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

x_train = torch.FloatTensor(np.random.uniform(0, np.pi, 1000)).view(-1,1) / np.pi
t_train = torch.FloatTensor(np.random.uniform(0, T, 1000)).view(-1,1) / T
x_train.requires_grad_(True)
t_train.requires_grad_(True)

def pinn_loss(model, x, t):
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return torch.mean((u_t - epsilon**2 * u_xx + u * (u**2 - 1))**2)

for epoch in range(1000):
    optimizer.zero_grad()
    loss = pinn_loss(model, x_train, t_train)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.2e}")

x_eval = torch.FloatTensor(x_vals).view(-1,1) / np.pi
t_eval = torch.FloatTensor([T]*Nx).view(-1,1) / T
with torch.no_grad():
    u_pinn = model(x_eval, t_eval).view(-1).numpy()
    u_pinn = np.clip(u_pinn, 0, 1)

# ==== MSE Evaluation ====
u_true = u_fd
u_theo = u_theorem(x_vals, T)
mse_theo = np.mean((u_true - u_theo)**2)
mse_spectral = np.mean((u_true - u_spectral)**2)
mse_pinn = np.mean((u_true - u_pinn)**2)

print(f"MSE Theorem-Based: {mse_theo:.2e}")
print(f"MSE Spectral:       {mse_spectral:.2e}")
print(f"MSE PINN:           {mse_pinn:.2e}")

# ==== Plot Comparison ====
plt.figure(figsize=(8,6))
plt.plot(x_vals, u_true, label='FD (Reference)', linewidth=2)
plt.plot(x_vals, u_theo, '--', label='Theorem-Based')
plt.plot(x_vals, u_spectral, '--', label='Spectral')
plt.plot(x_vals, u_pinn, '--', label='PINN')
plt.xlabel("x")
plt.ylabel("u(x, t=π)")
plt.title("Allen–Cahn Equation: Method Comparison at t=π")
plt.legend()
plt.tight_layout()
plt.savefig("allen_cahn_comparison.png", dpi=300)
plt.show()
