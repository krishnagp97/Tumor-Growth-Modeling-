import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

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
x0 = 1.0       # Initial tumor population
y0 = 5.0       # Initial immune population

# Time domain
t_min = 0.0
t_max = 50.0

# ------------------------------------------------------------
# Characteristic scales, used only to balance the two residual
# terms in the loss (the ODEs themselves are left untouched).
# Without this, the immune equation - whose terms are ~50x
# larger than the tumor equation's - dominates the gradient and
# the optimizer "gives up" on the tumor dynamics early.
# ------------------------------------------------------------
X_SCALE = 70.0
Y_SCALE = 250.0

# ============================================================
# PINN Model
# Input  : time t
# Output : tumor x(t), immune y(t)
#
# Two structural choices are baked into the forward pass:
#
# 1. HARD INITIAL CONDITION. Instead of an IC *loss* term, the
#    outputs are algebraically forced to satisfy x(0) = x0 and
#    y(0) = y0 for any network weights:
#
#       x(t) = exp( log(x0) + (1 - exp(-t)) * Nx(t) )
#       y(t) = y0 + (1 - exp(-t)) * Ny(t)
#
# 2. LOG-SPACE TUMOR OUTPUT. The tumor trajectory grows nearly
#    two orders of magnitude (1 -> ~70) before saturating - an
#    exponential-shaped curve that a tanh-MLP struggles to
#    reach in linear space. Modeling log(x) instead (then
#    exponentiating) matches the natural scale of the Gompertz
#    growth term m_C*x*ln(x/x_N) in the ODE itself, and turns
#    out to be the key fix: without it, gradient descent finds
#    it "cheaper" to collapse x toward 0 (a genuine but
#    unwanted low-residual branch of this system) than to climb
#    the correct exponential-like path.
# ============================================================

log_x0 = float(np.log(x0))
log_y0 = float(np.log(y0))

class PINN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 2)
        )

    def forward_raw(self, t):
        """Returns log(x(t)) and log(y(t)) (pre-exponentiation), so
        the loss function can build both residuals in log-space."""
        t_norm = t / t_max
        raw = self.network(t_norm)
        envelope = 1.0 - torch.exp(-t)

        log_x = log_x0 + envelope * raw[:, 0:1]
        log_y = log_y0 + envelope * raw[:, 1:2]
        return log_x, log_y

    def forward(self, t):
        log_x, log_y = self.forward_raw(t)
        return torch.exp(log_x), torch.exp(log_y)


model = PINN()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


def make_collocation(t_lo, t_hi, n_dense=350, n_sparse=150, dense_frac=0.4):
    """Collocation points biased toward the early, fast transient
    (roughly the first `dense_frac` of the current window), plus a
    uniform tail. Uniform-only sampling under-resolves the sharp
    tumor peak relative to the long flat tail, which is part of why
    the original network could ignore it."""
    dense_hi = t_lo + dense_frac * (t_hi - t_lo)
    t_dense = np.random.uniform(t_lo, dense_hi, n_dense)
    t_sparse = np.random.uniform(t_lo, t_hi, n_sparse)
    t_all = np.concatenate([[t_lo], t_dense, t_sparse, [t_hi]])
    t_all = np.sort(t_all)
    return torch.tensor(t_all, dtype=torch.float32).reshape(-1, 1)


def physics_loss_fn(t):
    t = t.clone().requires_grad_(True)

    log_x, log_y = model.forward_raw(t)
    x = torch.exp(log_x)
    y = torch.exp(log_y)

    dlogx_dt = torch.autograd.grad(
        log_x, t, grad_outputs=torch.ones_like(log_x), create_graph=True
    )[0]

    dlogy_dt = torch.autograd.grad(
        log_y, t, grad_outputs=torch.ones_like(log_y), create_graph=True
    )[0]

    # ------------------------------------------------------------
    # Both residuals are written in LOG SPACE (dividing each
    # original ODE through by its own state variable, valid since
    # x, y > 0 always in this model).
    #
    # Tumor:   d(log x)/dt = -m_C * log(x / x_N) - gamma * y
    # Immune:  d(log y)/dt =  m_I * (x - beta*x^2) - d + k / y
    #
    # This does two things the plain x/y-space residual doesn't:
    #  1. It removes the "collapse toward 0" shortcut for x (see
    #     note in the tumor-equation history above) - log(x/x_N)
    #     diverges as x -> 0, so the residual blows up instead of
    #     vanishing.
    #  2. It naturally balances the two equations' loss magnitudes
    #     without needing an ad-hoc scale constant. A fixed
    #     denominator (e.g. dividing the immune residual by 250)
    #     can hide a real, large *relative* error whenever y is
    #     small - which is exactly what let the immune curve
    #     collapse toward 0 in the first log-space attempt at this
    #     fix. Log-space residuals are scale-relative by
    #     construction, so no extra weighting is required.
    # ------------------------------------------------------------
    tumor_residual = (
        dlogx_dt
        + m_C * torch.log(x / x_N)
        + gamma * y
    )

    immune_residual = (
        dlogy_dt
        - m_I * (x - beta * x**2)
        + d
        - k / torch.clamp(y, min=1e-3)
    )

    loss = (
        torch.mean(tumor_residual ** 2)
        + torch.mean(immune_residual ** 2)
    )
    return loss


# ============================================================
# Curriculum ("time-marching") training
#
# Instead of asking the network to fit the whole t in [0, 50]
# window from a random initialization, we grow the window in
# stages. Early stages force the network to get the fast,
# high-curvature transient (t < ~15) right while the problem is
# still small; later stages extend that solution outward. This
# is what keeps training from collapsing onto the flat
# extinction branch, which is the easiest fit only when the
# whole 50-unit horizon is judged at once.
# ============================================================

stages = [
    {"t_hi": 15.0, "epochs": 4000},
    {"t_hi": 30.0, "epochs": 3000},
    {"t_hi": 50.0, "epochs": 5000},
]

history = []
epoch_counter = 0

for stage_idx, stage in enumerate(stages):
    t_hi = stage["t_hi"]
    n_epochs = stage["epochs"]

    for epoch in range(n_epochs):
        t_colloc = make_collocation(t_min, t_hi)

        optimizer.zero_grad()
        loss = physics_loss_fn(t_colloc)
        loss.backward()
        optimizer.step()

        epoch_counter += 1
        if epoch % 500 == 0:
            history.append((epoch_counter, loss.item()))
            print(
                f"Stage {stage_idx + 1}/{len(stages)} "
                f"(t in [0, {t_hi:.0f}])  "
                f"Epoch {epoch:5d} | Loss: {loss.item():.6f}"
            )

# ------------------------------------------------------------
# LBFGS fine-tuning on the full domain
#
# Adam gets us close; a second-order optimizer squeezes out the
# remaining residual error on the full [0, 50] window.
# ------------------------------------------------------------

lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=0.5,
    max_iter=500,
    history_size=50,
    line_search_fn="strong_wolfe",
)

t_full = make_collocation(t_min, t_max, n_dense=400, n_sparse=200)


def closure():
    lbfgs.zero_grad()
    loss = physics_loss_fn(t_full)
    loss.backward()
    return loss


lbfgs.step(closure)
final_loss = physics_loss_fn(t_full).item()
print(f"Final loss after LBFGS refinement: {final_loss:.6f}")

# ============================================================
# Prediction
# ============================================================

t_test = torch.linspace(
    t_min,
    t_max,
    500
).reshape(-1, 1)

with torch.no_grad():
    x_pred, y_pred = model(t_test)
    tumor_pred = x_pred.numpy().flatten()
    immune_pred = y_pred.numpy().flatten()

t_test = t_test.numpy().flatten()

# ============================================================
# Plot Tumor Population
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    t_test,
    tumor_pred,
    label="PINN Tumor"
)

plt.xlabel("Time")
plt.ylabel("Tumor Cell Density")
plt.title("PINN Solution - Tumor Dynamics")

plt.legend()
plt.grid()

plt.savefig(
    "11_pinn_tumor.png",
    dpi=300
)

plt.close()

# ============================================================
# Plot Immune Population
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    t_test,
    immune_pred,
    label="PINN Immune"
)

plt.xlabel("Time")
plt.ylabel("Immune Cell Density")
plt.title("PINN Solution - Immune Dynamics")

plt.legend()
plt.grid()

plt.savefig(
    "11_pinn_immune.png",
    dpi=300
)

plt.close()

# ============================================================
# Plot Both
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    t_test,
    tumor_pred,
    label="Tumor cells"
)

plt.plot(
    t_test,
    immune_pred,
    label="Immune cells"
)

plt.xlabel("Time")
plt.ylabel("Population")
plt.title("PINN Solution - Coupled Tumor-Immune Dynamics")

plt.legend()
plt.grid()

plt.savefig(
    "11_pinn_coupled.png",
    dpi=300
)

plt.close()

print("Saved: 11_pinn_tumor.png, 11_pinn_immune.png, 11_pinn_coupled.png")

# ============================================================
# Quantitative comparison against the RK45 reference solution
# ============================================================
from scipy.integrate import solve_ivp

def _ref_rhs(t, z):
    x, y = z
    dx = -m_C * x * np.log(x / x_N) - gamma * x * y
    dy = m_I * (x - beta * x**2) * y - d * y + k
    return [dx, dy]

_sol = solve_ivp(_ref_rhs, (t_min, t_max), [x0, y0], t_eval=t_test, method="RK45")
x_ref, y_ref = _sol.y[0], _sol.y[1]

x_rmse = np.sqrt(np.mean((tumor_pred - x_ref) ** 2))
y_rmse = np.sqrt(np.mean((immune_pred - y_ref) ** 2))
x_relerr = np.abs(tumor_pred - x_ref) / np.maximum(x_ref, 1e-6)
y_relerr = np.abs(immune_pred - y_ref) / np.maximum(y_ref, 1e-6)

print(f"Tumor  RMSE vs RK45: {x_rmse:.4f}   (max value ~{x_ref.max():.1f})")
print(f"Immune RMSE vs RK45: {y_rmse:.4f}   (max value ~{y_ref.max():.1f})")
print(f"Tumor  mean relative error:  {np.mean(x_relerr)*100:.2f}%")
print(f"Immune mean relative error:  {np.mean(y_relerr)*100:.2f}%")