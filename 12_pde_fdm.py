import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------- Parameters ----------------
m_C, x_N, D_x = 0.5, 100.0, 0.01
x0, epsilon = 1.0, 1e-4
L, T = 1.0, 50.0
N = 100
dr = L / N
r = np.linspace(0, L, N + 1)
t_eval = np.linspace(0, T, 201)

# np.trapezoid exists only in NumPy >= 2.0
trapz = getattr(np, "trapezoid", None) or np.trapz


def initial_condition(r):
    return epsilon + (x0 - epsilon) * np.sin(np.pi * r / L) ** 2


def laplacian(x):
    d2x = np.zeros_like(x)
    d2x[1:-1] = (x[2:] - 2 * x[1:-1] + x[:-2]) / dr**2
    d2x[0] = 2 * (x[1] - x[0]) / dr**2        # zero-flux
    d2x[-1] = 2 * (x[-2] - x[-1]) / dr**2     # zero-flux
    return d2x


def gompertz_pde(t, x):
    x_safe = np.maximum(x, 1e-12)
    return D_x * laplacian(x) + m_C * x * np.log(x_N / x_safe)


solution = solve_ivp(gompertz_pde, (0, T), initial_condition(r),
                     method="RK45", t_eval=t_eval, rtol=1e-6, atol=1e-8)
if not solution.success:
    raise RuntimeError(solution.message)

X = solution.y.T
print("Solution shape:", X.shape)
print("Min / Max density:", X.min(), X.max())

np.savez("fdm_solution.npz", r=r, t=solution.t, X=X)   # reusable by other scripts

# ---------------- Sanity check vs pure-ODE (no diffusion) ----------------
i_c = N // 2
x_ode = np.exp(np.log(x_N) + (np.log(initial_condition(r[i_c])) - np.log(x_N))
               * np.exp(-m_C * solution.t))
print("Max |FDM - ODE| at centre:", np.max(np.abs(X[:, i_c] - x_ode)))

# ---------------- Plots ----------------
plt.figure(figsize=(9, 5))
plt.pcolormesh(r, solution.t, X, shading="auto", cmap="viridis")
plt.colorbar(label="Tumor density")
plt.xlabel("Space (r)"); plt.ylabel("Time (t)")
plt.title("Tumor Gompertz PDE - FDM Solution")
plt.tight_layout(); plt.show()

plt.figure(figsize=(9, 5))
for time in [0, 1, 2, 5, 10, 20, 50]:
    k = np.argmin(np.abs(solution.t - time))
    plt.plot(r, X[k], label=f"t = {solution.t[k]:.1f}")
plt.xlabel("Space (r)"); plt.ylabel("Tumor density")
plt.title("Tumor Distribution at Different Times")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

plt.figure(figsize=(9, 5))
plt.plot(solution.t, X[:, i_c], label="FDM")
plt.plot(solution.t, x_ode, "--", label="ODE (no diffusion)")
plt.xlabel("Time (t)"); plt.ylabel("Tumor density")
plt.title("Tumor Growth at the Center")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

plt.figure(figsize=(9, 5))
plt.plot(solution.t, trapz(X, r, axis=1) / L)
plt.xlabel("Time (t)"); plt.ylabel("Average tumor density")
plt.title("Average Tumor Density Over Time")
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()