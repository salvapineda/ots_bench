import numpy as np
import pandas as pd
import gurobipy as gp
from scipy.sparse import csr_matrix

# ============================================================================
# Data Loading and Processing
# ============================================================================

def load_data(buses_file, branches_file, baseMVA=100):
    """Load buses and branches data from CSV files."""
    buses = pd.read_csv(buses_file, index_col="BUS_ID")
    branches = pd.read_csv(branches_file)
    return buses, branches, baseMVA


def build_incidence_matrix(buses, branches):
    """Build the node-arc incidence matrix A where:
    - A_nl = 1 if n is the from_bus of line l
    - A_nl = -1 if n is the to_bus of line l
    - A_nl = 0 otherwise
    """
    bus_id_list = buses.index.tolist()
    bus_id_to_idx = {bid: idx for idx, bid in enumerate(bus_id_list)}
    
    N = len(bus_id_list)
    L = len(branches)
    
    row_idx = []
    col_idx = []
    data = []
    
    for l, (_, row) in enumerate(branches.iterrows()):
        from_bus_idx = bus_id_to_idx[row["F_BUS"]]
        to_bus_idx = bus_id_to_idx[row["T_BUS"]]
        
        # From bus: +1
        row_idx.append(from_bus_idx)
        col_idx.append(l)
        data.append(1)
        
        # To bus: -1
        row_idx.append(to_bus_idx)
        col_idx.append(l)
        data.append(-1)
    
    A = csr_matrix((data, (row_idx, col_idx)), shape=(N, L))
    return A.toarray()


def ots_gurobi(buses_file, branches_file, baseMVA=100, time_limit=3600):
    """Solves the linearized DC Optimal Transmission Switching (DC-OTS) problem
    using Gurobi.
    
    Parameters:
    -----------
    buses_file : str
        Path to buses CSV file with columns: BUS_ID, PD, PMIN, PMAX, COST
    branches_file : str
        Path to branches CSV file with columns: F_BUS, T_BUS, BR_X, RATE_A, PFUPDC, PFLODC
    baseMVA : float
        Base MW value for scaling (default: 100)
    time_limit : float
        Time limit for Gurobi solver in seconds
    
    Returns:
    --------
    gp.Model
        Solved Gurobi model
    """
    
    # Load data from CSV files
    buses, branches, _ = load_data(buses_file, branches_file, baseMVA)
    
    # Use bus order as-is from CSV
    bus_id_list = buses.index.tolist()
    
    N = len(bus_id_list)  # Number of buses
    L = len(branches)  # Number of lines
    
    # ========================================================================
    # 1. Network Constants & Data Extraction
    # ========================================================================
    
    # Line parameters
    b = 1.0 / branches["BR_X"].values  # Susceptance b_l
    f_max = branches["RATE_A"].values  # \bar{f}_l (thermal capacity)
    f_min = -f_max  # \underline{f}_l
    
    # Big-M parameters
    M_up = branches["PFUPDC"].values  # \bar{M}_l
    M_lo = branches["PFLODC"].values  # \underline{M}_l
    
    # Bus parameters
    d = buses["PD"].values  # Demand d_n
    p_min = buses["PMIN"].values  # Minimum generation
    p_max = buses["PMAX"].values  # Maximum generation
    c = buses["COST"].values  # Linear marginal cost c_n
    
    # Build node-arc incidence matrix
    A = build_incidence_matrix(buses, branches)
    
    # ========================================================================
    # 2. Create Gurobi Model
    # ========================================================================
    m = gp.Model("dc_ots")
    m.setParam("TimeLimit", time_limit)
    
    # ========================================================================
    # 3. Decision Variables
    # ========================================================================
    p = m.addMVar(N, lb=p_min, ub=p_max, name="p")  # Generation at each bus
    theta = m.addMVar(N, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="theta")  # Bus angles
    f = m.addMVar(L, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="f")  # Line flows
    f_tilde = m.addMVar(L, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="f_tilde")  # Dummy flows
    x = m.addMVar(L, vtype=gp.GRB.BINARY, name="x")  # Line status (0 = off, 1 = on)
    
    # ========================================================================
    # 4. Objective Function: \min \sum c_n * p_n
    # ========================================================================
    m.setObjective(c @ p, gp.GRB.MINIMIZE)
    
    # ========================================================================
    # 5. Constraints
    # ========================================================================
    
    # Slack bus: \theta_1 = 0
    m.addConstr(theta[0] == 0, name="slack_bus")
    
    # Dummy Flow Definition: \tilde{f}_l = b_l * (\theta_n - \theta_m)
    # Using incidence matrix: f_tilde = b * (A.T @ theta)
    m.addConstr(f_tilde == b * (A.T @ theta), name="flow_dummy")
    
    # Big-M Flow Coupling: (1-x_l)*\underline{M}_l <= -f_l + \tilde{f}_l <= (1-x_l)*\bar{M}_l
    m.addConstr(-f + f_tilde <= M_up * (1 - x), name="bigM_upper")
    m.addConstr(-f + f_tilde >= M_lo * (1 - x), name="bigM_lower")
    
    # Thermal Capacity Limits: x_l*\underline{f}_l <= f_l <= x_l*\bar{f}_l
    m.addConstr(f <= f_max * x, name="thermal_upper")
    m.addConstr(f >= f_min * x, name="thermal_lower")
    
    # Node Power Balance: p_n - d_n = \sum flow_out - \sum flow_in
    m.addConstr(p - d == A @ f, name="power_balance")
    
    # ========================================================================
    # 6. Optimize
    # ========================================================================
    m.optimize()
    
    return m


if __name__ == "__main__":
    import os
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load data
    buses_file = os.path.join(script_dir, "buses.csv")
    branches_file = os.path.join(script_dir, "branches.csv")
    buses, branches, _ = load_data(buses_file, branches_file)
    
    print("Solving Optimal Transmission Switching (OTS) Problem...")
    model = ots_gurobi(buses_file, branches_file)
    
    # Print results
    if model.status == gp.GRB.OPTIMAL:
        print(f"\nMinimum cost: {model.objVal:.2f}")
        
        # Find lines that are switched OFF (x = 0)
        x_vars = [v for v in model.getVars() if v.VarName.startswith("x")]
        switched_off = [i for i, v in enumerate(x_vars) if v.X < 0.5]
        
        print(f"Lines switched off: {switched_off}")
    else:
        print(f"Model status: {model.status}")
        if model.status == gp.GRB.INFEASIBLE:
            print("The model is infeasible.")
        elif model.status == gp.GRB.TIME_LIMIT:
            print("Time limit reached.")