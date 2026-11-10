import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================
# Model Parameters
# ============================================================

m_C = 0.5
x_N = 100.0
gamma = 0.005

m_I = 0.01
beta = 0.001
d = 0.1
k = 0.5

# Initial conditions
x0 = 1.0
y0 = 5.0

# Time range
t_span = (0, 50)
t_eval = np.linspace(0, 50, 500)


# ============================================================
# Coupled Tumor–Immune ODE
# ============================================================

def tumor_immune_model(t, z):

    x, y = z

    # Tumor equation
    dx_dt = (
        -m_C * x * np.log(x / x_N)
        - gamma * x * y
    )

    # Immune equation
    dy_dt = (
        m_I * (x - beta * x**2) * y
        - d * y
        + k
    )

    return [dx_dt, dy_dt]


# ============================================================
# Numerical Solution using RK45
# ============================================================

solution = solve_ivp(
    tumor_immune_model,
    t_span,
    [x0, y0],
    t_eval=t_eval,
    method="RK45"
)

# Extract results
t = solution.t
tumor = solution.y[0]
immune = solution.y[1]


# ============================================================
# Tumor Population
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    t,
    tumor,
    label="Tumor cells"
)

plt.xlabel("Time")
plt.ylabel("Tumor Cell Density")
plt.title("Numerical Solution - Tumor Dynamics")

plt.legend()
plt.grid()

plt.savefig(
    "10_numerical_tumor.png",
    dpi=300
)

plt.show()


# ============================================================
# Immune Population
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    t,
    immune,
    label="Immune cells"
)

plt.xlabel("Time")
plt.ylabel("Immune Cell Density")
plt.title("Numerical Solution - Immune Dynamics")

plt.legend()
plt.grid()

plt.savefig(
    "10_numerical_immune.png",
    dpi=300
)

plt.show()


# ============================================================
# Coupled Dynamics
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    t,
    tumor,
    label="Tumor cells"
)

plt.plot(
    t,
    immune,
    label="Immune cells"
)

plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Numerical Solution - Coupled Tumor–Immune Dynamics")

plt.legend()
plt.grid()

plt.savefig(
    "10_numerical_coupled.png",
    dpi=300
)

plt.show()