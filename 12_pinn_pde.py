import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from pathlib import Path

# ==========================================
# 1. PARAMETERS
# ==========================================
m_C, x_N, D_x = 0.5, 100.0, 0.01
x0, epsilon = 1.0, 1e-4
L, T = 1.0, 50.0

N = 100
r = np.linspace(0, L, N + 1)
dr = L / N
t_eval = np.linspace(0, T, 201)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
np.random.seed(42)
print("Device:", device)

trapz = getattr(np, "trapezoid", None) or np.trapz

y_eps = epsilon / x_N
y_x0 = x0 / x_N


# ==========================================
# 2. FDM REFERENCE SOLUTION (tight tolerance)
# ==========================================
def initial_condition(r):
    return epsilon + (x0 - epsilon) * np.sin(np.pi * r / L) ** 2


def laplacian(x):
    d2x = np.zeros_like(x)
    d2x[1:-1] = (x[2:] - 2 * x[1:-1] + x[:-2]) / dr**2
    d2x[0] = 2 * (x[1] - x[0]) / dr**2        # Neumann
    d2x[-1] = 2 * (x[-2] - x[-1]) / dr**2     # Neumann
    return d2x


def gompertz_pde(t, x):
    x_safe = np.maximum(x, 1e-12)
    return D_x * laplacian(x) + m_C * x * np.log(x_N / x_safe)


print("\nComputing FDM reference solution...")
solution = solve_ivp(gompertz_pde, (0, T), initial_condition(r),
                     method="RK45", t_eval=t_eval, rtol=1e-10, atol=1e-12)
if not solution.success:
    raise RuntimeError(solution.message)

X_fdm = solution.y.T
R, TT = np.meshgrid(r, t_eval)

z_eval = torch.tensor(
    np.column_stack([R.flatten() / L, TT.flatten() / T]),
    dtype=torch.float64, device=device)

print("FDM solution shape:", X_fdm.shape)
print("FDM min/max:", X_fdm.min(), X_fdm.max())


# ==========================================
# 3. FDM DATASET SPLIT
# ==========================================
print("\nPreparing FDM dataset...")
rng = np.random.default_rng(42)

all_z = np.column_stack([R.flatten() / L, TT.flatten() / T])
all_y = (X_fdm / x_N).reshape(-1, 1)

indices = rng.permutation(len(all_z))
n_train = int(0.70 * len(indices))
n_val = int(0.15 * len(indices))
train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]


def make_tensor(data):
    return torch.tensor(data, dtype=torch.float64, device=device)


z_train, y_train = make_tensor(all_z[train_idx]), make_tensor(all_y[train_idx])
z_val, y_val = make_tensor(all_z[val_idx]), make_tensor(all_y[val_idx])
z_test, y_test = make_tensor(all_z[test_idx]), make_tensor(all_y[test_idx])

print("Total / train / val / test:",
      len(all_z), len(z_train), len(z_val), len(z_test))


# ==========================================
# 4. PINN MODEL (smooth, bounded, exact IC)
# ==========================================
class PINN(nn.Module):

    def __init__(self, width=96, depth=4, n_modes=6):
        super().__init__()
        self.n_modes = n_modes
        in_dim = n_modes + 6          # spatial modes + 6 time features

        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.network = nn.Sequential(*layers)

        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        r_ = z[:, 0:1]
        t_ = z[:, 1:2] * T

        # Initial condition
        y0 = y_eps + (y_x0 - y_eps) * torch.sin(np.pi * r_) ** 2

        # cos modes -> d/dr = 0 at r = 0 and r = 1 (Neumann BC)
        space = torch.cat(
            [torch.cos(k * np.pi * r_) for k in range(1, self.n_modes + 1)],
            dim=1)

        # multi-scale time features
        time = torch.cat([
            1 - torch.exp(-t_ / 1.0),
            1 - torch.exp(-t_ / 3.0),
            1 - torch.exp(-t_ / 6.0),
            1 - torch.exp(-t_ / 12.0),
            1 - torch.exp(-t_ / 25.0),
            t_ / T
        ], dim=1)

        u = self.network(torch.cat([space, time], dim=1))

        # env = 0 at t = 0 -> initial condition is exact
        env = 1.0 - torch.exp(-t_ / 1.0)

        # bounded in (0, 1): never exceeds carrying capacity
        return (1 - env) * y0 + env * torch.sigmoid(u)


model = PINN().to(device)


# ==========================================
# 5. PDE RESIDUAL
# ==========================================
def pde_residual(model, z):
    y = model(z)

    g = torch.autograd.grad(y, z, torch.ones_like(y), create_graph=True)[0]
    y_t = g[:, 1:2] / T

    y_rr = torch.autograd.grad(g[:, 0:1], z, torch.ones_like(y),
                               create_graph=True)[0][:, 0:1] / L**2

    growth = m_C * y * (-torch.log(torch.clamp(y, min=1e-8)))

    return 10.0 * (y_t - D_x * y_rr - growth)


# ==========================================
# 6. COLLOCATION POINT SAMPLING
# ==========================================
def sample_points(n, t_max):
    n_edge = n // 4

    r_u = torch.rand(n - n_edge, 1)
    r_e = 0.1 * torch.rand(n_edge, 1)
    flip = torch.rand(n_edge, 1) < 0.5
    r_e = torch.where(flip, 1.0 - r_e, r_e)

    r_pts = torch.cat([r_u, r_e], dim=0)

    # emphasise early times where dynamics are fastest
    tau = (t_max / T) * torch.rand(n, 1) ** 2.0

    return torch.cat([r_pts, tau], dim=1).to(device).requires_grad_(True)


# ==========================================
# 7. PREDICTION, METRICS, DATA WEIGHTS
# ==========================================
def predict():
    model.eval()
    with torch.no_grad():
        out = (x_N * model(z_eval)).cpu().numpy().reshape(len(t_eval), len(r))
    model.train()
    return np.maximum(out, 1e-12)


def rel_l2(Xp):
    return np.linalg.norm(X_fdm - Xp) / np.linalg.norm(X_fdm)


def val_error_fn():
    model.eval()
    with torch.no_grad():
        e = (torch.linalg.vector_norm(model(z_val) - y_val)
             / torch.linalg.vector_norm(y_val)).item()
    model.train()
    return e


def time_weight(z):
    """Extra data weight for the mid-time window t ~ 2-22."""
    t = z[:, 1:2] * T
    return 1.0 + 4.0 * torch.exp(-((t - 12.0) / 10.0) ** 2)


# ==========================================
# 8. DATA-ASSISTED PINN TRAINING (Adam)
# ==========================================
print("\nInitial relative L2 error:", f"{rel_l2(predict()):.6e}")

N_f = 1024
batch_size = 2048
epochs = 25000

lambda_pde = 1.0
lambda_data = 300.0

optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-5)

loss_history, pde_history, data_history = [], [], []
validation_history = []

best_error = float("inf")
best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

print("\nTraining Data-Assisted PINN (Adam)...\n")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # A. Physics loss
    z_f = sample_points(N_f, T)
    loss_pde = torch.mean(pde_residual(model, z_f) ** 2)

    # B. Data loss (weighted toward t ~ 2-22)
    batch_idx = torch.randint(0, len(z_train), (batch_size,), device=device)
    z_b = z_train[batch_idx]
    y_b = y_train[batch_idx]

    y_pred = model(z_b)
    loss_data = torch.mean(time_weight(z_b) * (y_pred - y_b) ** 2)

    # C. Combined loss
    loss = lambda_pde * loss_pde + lambda_data * loss_data
    loss.backward()
    optimizer.step()
    scheduler.step()

    loss_history.append(loss.item())
    pde_history.append(loss_pde.item())
    data_history.append(loss_data.item())

    # D. Validation
    if (epoch + 1) % 100 == 0:
        val_error = val_error_fn()
        validation_history.append((epoch + 1, val_error))

        if val_error < best_error:
            best_error = val_error
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1:5d} | "
                  f"Total: {loss.item():.4e} | "
                  f"PDE: {loss_pde.item():.4e} | "
                  f"Data: {loss_data.item():.4e} | "
                  f"Val: {val_error:.4e} | "
                  f"Best: {best_error:.4e}")

model.load_state_dict(best_state)
print("\nAdam stage done. Best validation relative L2:", f"{best_error:.6e}")


# ==========================================
# 9. L-BFGS POLISH (fixed collocation points)
# ==========================================
print("\nL-BFGS refinement (3 rounds)...")

torch.manual_seed(123)
z_f_fixed = sample_points(4096, T)
w_tr = time_weight(z_train)

lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=1.0, max_iter=2000,
    history_size=100, line_search_fn="strong_wolfe",
    tolerance_grad=1e-12, tolerance_change=1e-14)


def closure():
    lbfgs.zero_grad()
    lp = torch.mean(pde_residual(model, z_f_fixed) ** 2)
    ld = torch.mean(w_tr * (model(z_train) - y_train) ** 2)
    loss = lambda_pde * lp + lambda_data * ld
    loss.backward()
    return loss


for rnd in range(3):
    lbfgs.step(closure)
    e = val_error_fn()
    print(f"  Round {rnd + 1}: validation error = {e:.6e} "
          f"(best so far: {best_error:.6e})")
    if e < best_error:
        best_error = e
        best_state = {k: v.detach().clone()
                      for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
print("Final best validation relative L2 error:", f"{best_error:.6e}")


# ==========================================
# 10. INDEPENDENT TEST SET EVALUATION
# ==========================================
model.eval()
with torch.no_grad():
    test_pred = model(z_test)

test_relative_error = (torch.linalg.vector_norm(test_pred - y_test)
                       / torch.linalg.vector_norm(y_test)).item()
test_mae = torch.mean(torch.abs(test_pred - y_test)).item() * x_N
test_rmse = torch.sqrt(torch.mean((test_pred - y_test) ** 2)).item() * x_N

print("\n========== HELD-OUT TEST RESULTS ==========")
print(f"Test Relative L2 Error: {test_relative_error:.6e}")
print(f"Test MAE:               {test_mae:.6f}")
print(f"Test RMSE:              {test_rmse:.6f}")


# ==========================================
# 11. FULL-GRID RESULTS
# ==========================================
X_pinn = predict()
absolute_error = np.abs(X_fdm - X_pinn)

print("\n========== FULL-GRID RESULTS ==========")
print("FDM  min/max:", X_fdm.min(), X_fdm.max())
print("PINN min/max:", X_pinn.min(), X_pinn.max())
print(f"MAE:               {absolute_error.mean():.6f}")
print(f"RMSE:              {np.sqrt(np.mean(absolute_error**2)):.6f}")
print(f"Relative L2 error: {rel_l2(X_pinn):.6e}")
print(f"Maximum error:     {absolute_error.max():.6f}")

print("\n---- Relative L2 error at selected times ----")
for time in [0, 2, 5, 10, 20, 50]:
    k = np.argmin(np.abs(t_eval - time))
    e = np.linalg.norm(X_fdm[k] - X_pinn[k]) / np.linalg.norm(X_fdm[k])
    print(f"t = {t_eval[k]:5.1f}: {e:.3e}")


# ==========================================
# 12. AVERAGE TUMOR DENSITY
# ==========================================
avg_fdm = trapz(X_fdm, r, axis=1) / L
avg_pinn = trapz(X_pinn, r, axis=1) / L


# ==========================================
# 13. SAVE GRAPHS
# ==========================================
output_dir = Path("pde_graphs")
output_dir.mkdir(exist_ok=True)
DPI = 300
print("\nSaving graphs...")

# Graph 1: heatmaps
vmin = min(X_fdm.min(), X_pinn.min())
vmax = max(X_fdm.max(), X_pinn.max())

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
axes[0].pcolormesh(r, t_eval, X_fdm, shading="auto", cmap="viridis",
                   vmin=vmin, vmax=vmax)
im2 = axes[1].pcolormesh(r, t_eval, X_pinn, shading="auto", cmap="viridis",
                         vmin=vmin, vmax=vmax)
axes[0].set_title("FDM Solution", fontsize=14)
axes[1].set_title("Data-Assisted PINN Solution", fontsize=14)
for ax in axes:
    ax.set_xlabel("Space (r)")
    ax.set_ylabel("Time (t)")
fig.colorbar(im2, ax=axes, label="Tumor Density")
fig.savefig(output_dir / "01_FDM_vs_PINN_Heatmap.png", dpi=DPI, bbox_inches="tight")
plt.show()
plt.close(fig)

# Graph 2: profiles with an error row under each panel
fig, axes = plt.subplots(4, 3, figsize=(15, 13), constrained_layout=True,
                         gridspec_kw={"height_ratios": [3, 1, 3, 1]})
times = [0, 2, 5, 10, 20, 50]

for i, time in enumerate(times):
    k = np.argmin(np.abs(t_eval - time))
    row, col = (0, i) if i < 3 else (2, i - 3)
    ax, axe = axes[row, col], axes[row + 1, col]

    ax.plot(r, X_fdm[k], label="FDM", linewidth=2)
    ax.plot(r, X_pinn[k], "--", label="Data-Assisted PINN", linewidth=2)
    ax.set_title(f"Time = {t_eval[k]:.1f}")
    ax.set_ylabel("Tumor Density")
    ax.ticklabel_format(useOffset=False, axis="y")

    # pad the axis so tiny differences are not blown up
    lo, hi = X_fdm[k].min(), X_fdm[k].max()
    pad = max(0.02 * hi, 0.5 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)
    ax.legend()
    ax.grid(alpha=0.3)

    axe.plot(r, np.abs(X_fdm[k] - X_pinn[k]), color="crimson")
    axe.set_ylabel("|error|")
    axe.set_xlabel("Space (r)")
    axe.grid(alpha=0.3)

fig.suptitle("Tumor Distribution: FDM vs Data-Assisted PINN", fontsize=16)
fig.savefig(output_dir / "02_Tumor_Distribution.png", dpi=DPI, bbox_inches="tight")
plt.show()
plt.close(fig)

# Graph 3: centre density
c = N // 2
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(t_eval, X_fdm[:, c], label="FDM", linewidth=2)
ax.plot(t_eval, X_pinn[:, c], "--", label="Data-Assisted PINN", linewidth=2)
ax.set_xlabel("Time (t)")
ax.set_ylabel("Tumor Density")
ax.set_title("Center Tumor Density: FDM vs PINN")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "03_Center_Tumor_Growth.png", dpi=DPI, bbox_inches="tight")
plt.show()
plt.close(fig)

# Graph 4: average density
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(t_eval, avg_fdm, label="FDM", linewidth=2)
ax.plot(t_eval, avg_pinn, "--", label="Data-Assisted PINN", linewidth=2)
ax.set_xlabel("Time (t)")
ax.set_ylabel("Average Tumor Density")
ax.set_title("Average Tumor Density: FDM vs PINN")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "04_Average_Tumor_Density.png", dpi=DPI, bbox_inches="tight")
plt.show()
plt.close(fig)

# Graph 5: absolute error heatmap
fig, ax = plt.subplots(figsize=(9, 5))
im = ax.pcolormesh(r, t_eval, absolute_error, shading="auto", cmap="magma")
fig.colorbar(im, ax=ax, label="Absolute Error")
ax.set_xlabel("Space (r)")
ax.set_ylabel("Time (t)")
ax.set_title("Absolute Error: |FDM - PINN|")
fig.tight_layout()
fig.savefig(output_dir / "05_Absolute_Error_Heatmap.png", dpi=DPI, bbox_inches="tight")
plt.show()
plt.close(fig)

# Graph 6: training loss
fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogy(np.maximum(loss_history, 1e-16), label="Total Loss", linewidth=1.2)
ax.semilogy(np.maximum(pde_history, 1e-16), label="PDE Loss", linewidth=1.0)
ax.semilogy(np.maximum(data_history, 1e-16), label="Data Loss", linewidth=1.0)
ax.set_xlabel("Training Epoch")
ax.set_ylabel("Loss")
ax.set_title("Data-Assisted PINN Training Loss (Adam stage)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "06_PINN_Training_Loss.png", dpi=DPI, bbox_inches="tight")
plt.show()
plt.close(fig)


# ==========================================
# 14. SAVE NUMERICAL RESULTS
# ==========================================
np.savez(
    output_dir / "comparison_results.npz",
    r=r, t=t_eval,
    X_fdm=X_fdm, X_pinn=X_pinn,
    absolute_error=absolute_error,
    avg_fdm=avg_fdm, avg_pinn=avg_pinn,
    loss_history=np.asarray(loss_history),
    pde_history=np.asarray(pde_history),
    data_history=np.asarray(data_history),
    validation_history=np.asarray(validation_history),
    best_validation_error=best_error,
    test_relative_error=test_relative_error,
    test_mae=test_mae,
    test_rmse=test_rmse
)

torch.save(model.state_dict(), output_dir / "pinn_best.pt")


# ==========================================
# 15. COMPLETION
# ==========================================
print("\n========== GRAPH SAVING COMPLETED ==========")
print("Output folder:", output_dir.resolve())
for file in sorted(output_dir.iterdir()):
    print(file.name)
print("\nAll graphs and numerical results saved successfully!")