# 🔬 PDE Solvers and RDSF-Based Parameter Estimation

This repository provides a collection of Python scripts for analyzing and solving well-known partial differential equations (PDEs) using multiple methods. The project focuses on:

- Developing and comparing numerical solvers: Finite Difference, Spectral Methods, and PINNs
- Designing trial solutions and residual functions
- Estimating optimal parameters using the RDSF algorithm (Residual-Driven Search for Function parameters)

## ✅ Included PDEs

- Allen–Cahn Equation
- Burgers' Equation
- Fisher–KPP Equation
- 1D Navier–Stokes Equation

Each PDE is solved using multiple numerical approaches, and residual-based heatmaps are used to tune trial function parameters.

---

## 📂 File Overview

| File Name | Description |
|----------|-------------|
| `allen_cahn_method_comparison.py` | Solves the Allen–Cahn equation using Theorem-based trial, Finite Difference, and PINN methods. |
| `allen_cahn_rdsf_sweep.py` | Applies RDSF algorithm to find optimal ε and δ values for minimizing residuals in the Allen–Cahn equation. |
| `burgers_equation_method_comparison.py` | Compares Theorem-based, Spectral, Finite Difference, and PINN solutions for Burgers' equation. |
| `burgers_rdsf_residual_sweep.py` | Performs residual error sweep for Burgers’ equation using the RDSF method and visualizes optimal parameters. |
| `fisher_kpp_compare_methods.py` | Compares multiple solvers for the Fisher–KPP equation at \( t = \pi \), including Theorem-based, FD, Spectral, and PINN. |
| `fisher_kpp_rdsf_sweep.py` | Runs a residual-based parameter sweep over ε and δ for the Fisher–KPP equation using RDSF. |
| `navier_stokes_1d_compare_methods.py` | Compares Theorem-based, FD, Spectral, and PINN methods for solving the 1D Navier–Stokes equation. |
| `navier_stokes_rdsf_symbolic_residual_sweep.py` | Uses SymPy for symbolic computation of residuals in Navier–Stokes and performs RDSF over ε and δ. |

---

## 📈 Features

- Automatic generation of heatmaps for residuals
- Calculation of MSE (Mean Squared Error) between solvers
- Support for symbolic differentiation (SymPy) and automatic differentiation (PyTorch)
- Visualization and comparison of solver outputs at specific time snapshots (e.g., \( t = \pi \))

---

## 🔧 Installation

Install the required dependencies using pip:

```bash
pip install numpy matplotlib scipy torch sympy tqdm seaborn
