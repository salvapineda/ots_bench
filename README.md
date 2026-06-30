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

Consider a power system represented by a graph with nodes $\mathcal{N}$ and transmission lines $\mathcal{L}$. Each node $n \in \mathcal{N}$ has a demand $d_n$ and a generator producing power $p_n$ within limits $[\underline{p}_n, \overline{p}_n]$ at a marginal cost $c_n$. 

Each line $l = (n,m) \in \mathcal{L}$ has a susceptance $b_l$ and thermal capacity limits $[\underline{f}_l, \overline{f}_l]$. The operational status of each line is modeled by a binary variable $x_l \in \{0,1\}$, where $x_l = 1$ if the line is active and $x_l = 0$ if it is disconnected. We introduce a dummy variable $\tilde{f}_l$ to capture the unconstrained physical power flow dictated by the bus voltage angles $\theta_n$ and $\theta_m$. 

The non-linear, mixed-integer DC-OTS problem is formulated as follows:

\begin{subequations}\label{eq:OTS_NP}
\begin{IEEEeqnarray}{l}
\min_{p, f, \tilde{f}, \theta, x} \quad \sum_{n \in \mathcal{N}} c_{n} \, p_{n} \label{eq:OTS_NP_obj}\\
\text{subject to}  \nonumber \\ 
f_l = x_l \tilde{f}_l, \quad \forall l \in \mathcal{L} \label{eq:OTS_NP_Flow}\\
\underline{f}_l \leq f_l \leq \overline{f}_l, \quad \forall l \in \mathcal{L} \label{eq:OTS_NP_Flow_limit_S}\\
\tilde{f}_l = b_l(\theta_n-\theta_m), \quad \forall l=(n,m) \in \mathcal{L} \label{eq:OTS_NP_Flow_Dummy}\\
p_n - d_n = \sum_{l\in\mathcal{L}(n,\cdot)} f_l - \sum_{l\in\mathcal{L}(\cdot,n)} f_l, \quad \forall n \in \mathcal{N} \label{eq:OTS_NP_PB}\\
\underline{p}_n \leq p_n \leq \overline{p}_n, \quad \forall n \in \mathcal{N} \label{eq:OTS_NP_Plimits}\\
\theta_1 = 0 \label{eq:OTS_NP_slack}\\
x_l \in \{0,1\}, \quad \forall l \in \mathcal{L} \label{eq:OTS_NP_binary}
\end{IEEEeqnarray}
\end{subequations}

> **Note on LLM Complexity:** The non-linear product $x_l \tilde{f}_l$ in \eqref{eq:OTS_NP_Flow} represents a classic disjunctive logic constraint. In the accompanying solver code, this is linearized using the **Big-M method**. Correctly translating this non-linear conceptual formulation into a stable linear MILP is a primary challenge where frontier LLMs frequently fail.

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
