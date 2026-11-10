## ->Introduction to Cancer

Cancer is a group of diseases characterized by the uncontrolled growth and spread of abnormal cells in the body. Under normal conditions, cells grow, divide, and die in a regulated manner. However, in cancer, this regulation breaks down, leading to the formation of a mass of abnormal cells called a tumor.

## ->What causes cancer?

Cancer occurs due to genetic mutations that affect how cells behave. These mutations may be caused by:

Environmental factors (pollution, radiation, chemicals)
Lifestyle factors (smoking, diet)
Infections (certain viruses)
Inherited genetic defects

These mutations disrupt normal cellular processes such as:

Cell division
DNA repair
Programmed cell death (apoptosis) 

## -> Types of tumors

Tumors can be classified into two main types:

Benign Tumors
Non-cancerous
Do not spread to other parts of the body

Premalignant (Pre-cancerous) Tumors
Not cancer yet, but can become cancer over time
Early detection can prevent cancer


Malignant Tumors
Cancerous
Can invade nearby tissues
Can spread to distant organs (metastasis)

## -> Tumor Growth Dynamics

Tumor growth is not linear. It follows specific biological patterns:

Initial Phase: Slow growth due to limited cells
Rapid Growth Phase: Exponential increase in cell number
Saturation Phase: Growth slows due to lack of nutrients and space

This behavior is often modeled using mathematical equations like:

Exponential growth model
Logistic growth model
Gompertz model (most realistic)

## -> Why model cancer mathematically?

Mathematical modeling helps in:

Understanding tumor growth behavior
Predicting future growth
Evaluating treatment strategies
Simulating biological processes

## 1. Exponential Growth Model
   dN/dt = rN

Problem:
Assumes unlimited growth
Tumors would grow forever without slowing

## 2. Logistic Model

dN/dt = rN(1-N/K)

Problem:
Curve is symmetric S-shape
Tumor growth is NOT symmetric
Real tumors:
grow fast early
slow down gradually (not evenly)

## 3. Gompertz Model

dN/dt = rN ln(K/N)

The Gompertz ODE (Gompertz differential equation) is a mathematical model used to describe growth that starts fast and then slows down over time.

advantages:
Asymmetric growth curve
Strong early growth, slow long-term saturation
Matches real tumor data from experiments

Widely used in:
oncology research
drug response modeling

We first solve using numerical methods to obtain a reliable reference solution, then apply PINN to learn the governing physics and compare accuracy


## References

1. National Cancer Institute. "What is Cancer?"  
2. World Health Organization. "Cancer Overview"  
3. J. D. Murray, *Mathematical Biology*  
4. Leah Edelstein-Keshet, *Mathematical Models in Biology*  
5. Gompertz Law of Growth  
6. Raissi et al. (2019), Physics-Informed Neural Networks
