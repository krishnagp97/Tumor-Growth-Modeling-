import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Gompertz ODE
def gompertz(t, N):
    a = 0.5
    K = 100
    return -a * N * np.log(N / K)

# time range
t_span = (0, 10)
t_eval = np.linspace(0, 10, 100)

# initial tumor size
N0 = [1]

# solve ODE Uses numerical method (Runge-Kutta)
sol = solve_ivp(gompertz, t_span, N0, t_eval=t_eval)

# plot result
plt.plot(sol.t, sol.y[0], label="Tumor Growth")

plt.xlabel("Time")
plt.ylabel("Tumor Size")
plt.title("Gompertz Cancer Growth Model")
plt.legend()

plt.savefig("output.png")
plt.show()
