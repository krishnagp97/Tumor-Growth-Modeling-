import numpy as np
import matplotlib.pyplot as plt

# Parameters
K = 1000      # Carrying capacity
N0 = 50       # Initial tumor size
a = 0.2       # Growth rate

# Time values
t = np.linspace(0, 50, 500)

# Analytical solution of Gompertz Model
N = K * np.exp(np.log(N0 / K) * np.exp(-a * t))

# Plotting
plt.figure(figsize=(8,5))
plt.plot(t, N, linewidth=2)

# Labels and title
plt.title("Tumor Growth using Gompertz Analytical Solution")
plt.xlabel("Time")
plt.ylabel("Tumor Size N(t)")
plt.grid(True)

# Show graph
plt.savefig("analytic.png", dpi=300)
plt.show()