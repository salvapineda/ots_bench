# OTS-Bench: Optimal Transmission Switching for LLM Optimization Modeling

This repository contains the benchmark data, mathematical formulations, and solver code for the **Optimal Transmission Switching (OTS)** problem, contributed to the community-driven benchmark initiative for evaluating Large Language Models (LLMs) on optimization modeling (`OR-Bench`).

---

## 1. Problem Descriptions

### Vague, Business-Level Description
> *"We are running into high operational costs across our electricity transmission grid during peak hours due to congestion. A colleague suggested that we could actually save money on generation costs by strategically opening up certain circuit breakers and taking specific transmission lines entirely offline. This sounds completely backwards—how does removing a power line lower costs? We need a decision-making tool that looks at our current grid topology, line capacities, and generator costs, and tells us exactly which lines to switch off to minimize our total system cost without causing blackouts or violating basic electrical physics."*

### Precise Technical Description
The Optimal Transmission Switching (OTS) problem aims to minimize the total generation cost in an electricity network over a specific time horizon by optimizing both generator outputs and network topology (line switching statuses). 

Given a network graph $\mathcal{G} = (\mathcal{N}, \mathcal{L})$ where $\mathcal{N}$ is the set of buses and $\mathcal{L}$ is the set of transmission lines:
* **Objective:** Minimize total active power generation cost.
* **Decision Variables:** Continuous generation outputs ($P_{g}$), continuous bus voltage angles ($\theta_i$), and binary line status variables ($z_l \in \{0,1\}$), where $z_l = 0$ indicates a line is switched off.
* **Constraints:** DC power flow equations, line thermal capacity limits, generation bounds, and bus voltage angle differences. 

To handle the conditional power flow when a line is opened ($z_l = 0$), the disjunctive constraints are formulated using the **Big-M optimization method**. This creates a computationally challenging Mixed-Integer Linear Program (MILP) that requires careful parameter tuning to avoid numerical instability.

---

## 2. Mathematical Formulation

The standard DC-OTS formulation implemented here is as follows:

$$\min_{P_g, \theta, z} \sum_{g \in \mathcal{G}} C_g(P_g)$$

Subject to:
* **Bus Power Balance:**
  $$\sum_{g \in \mathcal{G}_i} P_g - P_{d,i} = \sum_{l \in \mathcal{L}_i^{out}} P_l - \sum_{l \in \mathcal{L}_i^{in}} P_l \quad \forall i \in \mathcal{N}$$

* **Line Flow (Big-M Disjunction):**
  $$-M_l (1 - z_l) \le P_l - B_l(\theta_i - \theta_j) \le M_l (1 - z_l) \quad \forall l=(i,j) \in \mathcal{L}$$

* **Thermal Capacity Limits:**
  $$-F_l^{max} z_l \le P_l \le F_l^{max} z_l \quad \forall l \in \mathcal{L}$$

* **Variable Bounds:**
  $$P_g^{min} \le P_g \le P_g^{max} \quad \forall g \in \mathcal{G}$$
  $$z_l \in \{0, 1\} \quad \forall l \in \mathcal{L}$$

---

## 3. Repository Structure & Artifacts

* `/data`: Contains `.csv` files representing standard grid topologies (e.g., modified IEEE test cases) containing bus data, line parameters ($B_l, F_l^{max}$), and generator cost coefficients ($C_g$).
* `/src`: Contains the Python implementation utilizing **Gurobi** to solve the MILP formulation.

---

## 4. Getting Started

### Prerequisites
* Python 3.10+
* Gurobi Optimizer (with a valid license installed)
* `pandas`, `gurobipy`

### Installation
```bash
git clone [https://github.com/your-username/ots_bench.git](https://github.com/your-username/ots_bench.git)
cd ots_bench
pip install -r requirements.txt
