import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# parameters
#a → tumor growth rate
#K → maximum tumor size (carrying capacity)

a = 0.5
K = 100

# neural network
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
        return K * torch.sigmoid(self.net(t))   # bounded between 0 and K

model = PINN()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training points
t = torch.linspace(0, 10, 100).reshape(-1, 1)
t.requires_grad = True

# initial condition
t0 = torch.tensor([[0.0]])
N0 = torch.tensor([[1.0]])

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

    # strong initial condition
    N_pred0 = model(t0)
    ic_loss = (N_pred0 - N0)**2 * 10   # weighted

    loss = torch.mean(physics**2) + ic_loss

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# prediction
t_test = torch.linspace(0, 10, 100).reshape(-1, 1)
N_pred = model(t_test).detach().numpy()

# plot
plt.plot(t_test.numpy(), N_pred, label="PINN Solution")

plt.xlabel("Time")
plt.ylabel("Tumor Size")
plt.title("Fixed PINN Tumor Growth")
plt.legend()

plt.savefig("pinn_fixed.png", dpi=300)
plt.show()