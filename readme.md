# MgNet for Steady-State Radiative Transfer Equation

This repository provides the Python implementation of **MgNet** for solving the **steady-state radiative transfer equation (RTE)**. In this project, MgNet is trained and used as a **preconditioner** for the fully discretized linear system arising from the steady-state RTE.

The numerical discretization is based on:

- **DOM** (Discrete Ordinates Method) for angular discretization
- **ATFPS** for the discretization scheme used in the loss construction and linear system formulation

---

## Project Overview

The repository contains:

- data generation scripts for different physical regimes,
- model definitions for **CoeffNet** and **MgNet**,
- training code for MgNet,
- testing scripts for evaluating preconditioning performance,
- GMRES implementation in PyTorch,
- plotting utilities for visualization.

The main goal is to compare the performance of the following preconditioners for solving the fully discretized steady-state RTE system:

1. **Classical block Jacobi preconditioner**
2. **MgNet preconditioner trained with 10,000 samples**
3. **MgNet preconditioner trained with 20,000 samples**

---

## File Folder Description

### `discretized_parameters`

The dictionary `discretized_parameters` stores the following information:

1. **`quadrature`**  
   Discretized angular/velocity directions and their corresponding quadrature weights from DOM.

2. **`Kappa`**  
   The discretized kernel function.

3. **Dataset information for different physical regimes**, including:
   - `Coef`
   - `fsmLRBTC`
   - `I2A`
   - `MLRBT`
   - `VecSize`

   The last four items are generated from the **ATFPS discretization scheme**. They are derived from `Coef` and are prepared for the design of the loss function.

   > The dataset can be generated using `RTE_datagenerator.py`.

---

### `model_precond`, `train_loss`, and `val_loss`

These dictionaries store:

- the trained **MgNet model** (viewed as a preconditioner),
- the **training loss**,
- the **validation loss**

for each dataset/regime.

---

## File Description

### `train.py`
Training script for MgNet.

### `test.ipynb`
This notebook:

1. generates the test dataset,
2. visualizes the generated data,
3. applies **GMRES** with three different preconditioners to solve the fully discretized linear system from the steady-state RTE.

The three preconditioners are:

- classical block Jacobi preconditioner,
- MgNet trained with **10,000** samples,
- MgNet trained with **20,000** samples.

---

### `RTE_datagenerator.py`
Generates datasets for different physical regimes. The generated data include:

- `Coef`
- `fsmLRBTC`
- `I2A`
- `MLRBT`
- `VecSize`

The last four quantities are generated from `Coef` through the **ATFPS discretization scheme** and are used in the design of the loss function.

---

### `RTE_dataloader.py`
Wraps the generated data into a custom dataset so that it can be used conveniently with tools such as `DataLoader`.

---

### `RTE_mgmodel.py`
Defines and constructs **CoeffNet** and **MgNet**.

---

### `RTE_ATFPS_func.py`
Implements the construction of the loss function.

Key points:

- The loss is **physics-informed**.
- It uses the residual of the fully discretized system based on **TFPS/ATFPS**.
- The degree of compression in the loss function is automatically controlled by the threshold `tol_exp`.
- If `tol_exp` is very large, then no compression is applied.
- The compression behavior also depends on the optical properties of the background medium.

---

### `gmres.py`
PyTorch implementation of the **GMRES** algorithm.

---

### `plot.py`
Provides utilities for plotting 2D images of:

- angular flux,
- scalar flux.

---

### `utils.py`
Contains common utility functions, including:

- saving models,
- saving training/validation losses,
- checking Chinese font support,
- and other helper functions.

---

### `someplot.ipynb`
Contains additional plotting code for producing cleaner and more visually appealing training/validation loss figures.

---

## Usage

### 1. Train MgNet

Run `train.py` for each physical regime (for example: `transport`, `diffusion`, `interface`, `bufferzone`, etc.).

Recommended settings:

- `num_coef = 1000`, `num_data = 10000`
- `num_coef = 2000`, `num_data = 20000`

That is, for each regime, you may train two MgNet models:

- one with **10,000 samples**,
- one with **20,000 samples**.

---

### 2. Generate data if needed

If the required `RTEData` in `train.py` does not exist, use:

```bash
python RTE_datagenerator.py
