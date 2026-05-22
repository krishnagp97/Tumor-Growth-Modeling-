# Tumor Growth Modeling using Gompertz ODE and Physics-Informed Neural Networks (PINNs)

## Introduction to Cancer

Cancer is a group of diseases characterized by the uncontrolled growth and spread of abnormal cells in the body. Under normal conditions, cells grow, divide, and die in a regulated manner. In cancer, this regulation breaks down, leading to the formation of abnormal masses of cells called tumors.

---

## What Causes Cancer?

Cancer occurs due to genetic mutations that affect normal cellular behavior. These mutations may arise from:

- Environmental factors (pollution, radiation, chemicals)
- Lifestyle factors (smoking, diet)
- Infections (certain viruses)
- Inherited genetic defects

These mutations disrupt essential biological processes such as:

- Cell division
- DNA repair
- Programmed cell death (apoptosis)

---

## Types of Tumors

### Benign Tumors
- Non-cancerous
- Do not spread to other parts of the body

### Premalignant (Pre-cancerous) Tumors
- Not yet cancerous
- Can develop into cancer over time
- Early detection may prevent malignancy

### Malignant Tumors
- Cancerous
- Can invade nearby tissues
- Can spread to distant organs (metastasis)

---

## Tumor Growth Dynamics

Tumor growth is not linear and follows distinct biological phases:

1. **Initial Phase**  
   Slow growth due to a small number of cells.

2. **Rapid Growth Phase**  
   Exponential increase in tumor cell population.

3. **Saturation Phase**  
   Growth slows because of limited nutrients and space.

Mathematical models help describe these growth patterns.

---

# Mathematical Models for Tumor Growth

## 1. Exponential Growth Model

\[
\frac{dN}{dt}=rN
\]

### Limitation
- Assumes unlimited growth
- Ignores environmental constraints
- Unrealistic for long-term tumor behavior

---

## 2. Logistic Growth Model

\[
\frac{dN}{dt}=rN\left(1-\frac{N}{K}\right)
\]

### Limitation
- Produces a symmetric S-shaped curve
- Real tumor growth is generally asymmetric
- Tumors grow rapidly early and slow gradually later

---

## 3. Gompertz Growth Model

\[
\frac{dN}{dt}=rN\ln\left(\frac{K}{N}\right)
\]

The Gompertz differential equation is widely used to model biological growth processes that initially grow rapidly and gradually saturate over time.

### Advantages
- Produces an asymmetric growth curve
- Captures strong early growth and slow long-term saturation
- Matches experimental tumor growth data more accurately

### Applications
- Oncology research
- Drug response modeling
- Tumor progression studies

---

# Why Mathematical Modeling?

Mathematical modeling helps researchers:

- Understand tumor growth behavior
- Predict future tumor progression
- Evaluate treatment strategies
- Simulate biological processes computationally

---

# Project Objective

This project focuses on solving the Gompertz tumor growth model using:

1. Analytical Solution
2. Numerical Solver (RK45)
3. Physics-Informed Neural Networks (PINNs)

The analytical solution is treated as the ground truth for validation. RK45 provides a numerical approximation, while the PINN learns the governing physics directly from the differential equation and initial conditions.

The performance of RK45 and PINN solutions is compared against the analytical solution to evaluate accuracy and learning capability.

---

# Physics-Informed Neural Networks (PINNs)

Physics-Informed Neural Networks (PINNs) are neural networks trained using physical laws represented by differential equations.

Instead of relying only on data, PINNs incorporate:

- Governing differential equations
- Initial conditions
- Boundary conditions

This enables the network to learn physically consistent solutions.

---

# Problem Statement

Given the Gompertz ODE governing tumor growth, efficiently learn its solution using a Physics-Informed Neural Network that simultaneously:

- Satisfies the governing differential equation
- Obeys the initial condition
- Approximates the analytical solution accurately

---

# Methodology

1. Define the Gompertz differential equation
2. Derive the analytical solution
3. Solve numerically using RK45
4. Train the PINN model
5. Compare:
   - Analytical vs RK45
   - Analytical vs PINN
   - RK45 vs PINN

---

# Technologies Used

- Python
- PyTorch / TensorFlow
- NumPy
- SciPy
- Matplotlib

---

# Results

The project demonstrates that PINNs can successfully approximate tumor growth dynamics governed by the Gompertz equation while maintaining consistency with the underlying physics.

---

# References

1. National Cancer Institute — *What is Cancer?*  
2. World Health Organization — *Cancer Overview*  
3. J. D. Murray — *Mathematical Biology*  
4. Leah Edelstein-Keshet — *Mathematical Models in Biology*  
5. Gompertz Law of Growth  
6. Raissi et al. (2019) — *Physics-Informed Neural Networks*
