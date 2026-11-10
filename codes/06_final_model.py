import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Parameters

a = 0.5
K = 100


# Numerical Solution (Baseline)

def gompertz(t, N):
    return -a * N * np.log(N / K)

t_span = (0, 10)
t_eval = np.linspace(0, 10, 100)
N0 = [1]

sol = solve_ivp(gompertz, t_span, N0, t_eval=t_eval)
t_np = sol.t
N_true = sol.y[0]


# PINN Model

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t):
        return K * torch.sigmoid(self.net(t))   # bounded output

model = PINN()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training points
t = torch.tensor(t_np, dtype=torch.float32).reshape(-1, 1)
t.requires_grad = True

# initial condition
t0 = torch.tensor([[0.0]])
N0_torch = torch.tensor([[1.0]])

# convert true data (for guidance)
N_true_torch = torch.tensor(N_true, dtype=torch.float32).reshape(-1, 1)


# Training

for epoch in range(6000):
    optimizer.zero_grad()

    N = model(t)

    dN_dt = torch.autograd.grad(
        N, t,
        grad_outputs=torch.ones_like(N),
        create_graph=True
    )[0]

    # physics loss
    physics = dN_dt + a * N * torch.log(N / K + 1e-6)

    # initial condition
    ic_loss = (model(t0) - N0_torch)**2 * 50

    # data loss (VERY IMPORTANT FIX)
    data_loss = (N - N_true_torch)**2

    # total loss
    loss = torch.mean(physics**2) + torch.mean(data_loss) + ic_loss

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# Prediction

t_test = torch.tensor(t_np, dtype=torch.float32).reshape(-1, 1)
N_pinn = model(t_test).detach().numpy()


# Plot Comparison

plt.plot(t_np, N_true, label="Numerical Solution")
plt.plot(t_np, N_pinn, '--', label="PINN Solution")

plt.xlabel("Time")
plt.ylabel("Tumor Size")
plt.title("Gompertz Model: Numerical vs PINN")
plt.legend()

plt.savefig("comparison.png", dpi=300)
plt.show()