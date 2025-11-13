import numpy as np
import numpy.linalg as la
import pandas as pd
import csv
import scipy.io as spio
import matplotlib
matplotlib.use('Agg')
import os
import sys
import argparse
import re
from juliacall import Main as jl
import gurobipy as gp
from gurobipy import GRB

# Solvers
import gurobipy as gp

# Maros Meszaros optimal objective values
from maros_meszaros import OPT_COST_MAP

# Solver settings
TIME_LIMIT = 1000.
EPS_LOW = 1e-03
EPS_HIGH = 1e-06

# Solvers
GUROBI = 'GUROBI'
GUROBI_high = GUROBI + "_high"
pdqp = 'pdqp'
pdqp_high = pdqp + '_high'


SETTINGS = {
    GUROBI: {'TimeLimit': TIME_LIMIT,
             'FeasibilityTol': EPS_LOW,
             'OptimalityTol': EPS_LOW,
             },
    GUROBI_high: {'TimeLimit': TIME_LIMIT,
                  'FeasibilityTol': EPS_HIGH,
                  'OptimalityTol': EPS_HIGH,
                  },
}

# Solver constants
OPTIMAL = "optimal"
OPTIMAL_INACCURATE = "optimal inaccurate"
PRIMAL_INFEASIBLE = "primal infeasible"
PRIMAL_INFEASIBLE_INACCURATE = "primal infeasible inaccurate"
DUAL_INFEASIBLE = "dual infeasible"
DUAL_INFEASIBLE_INACCURATE = "dual infeasible inaccurate"
PRIMAL_OR_DUAL_INFEASIBLE = "primal or dual infeasible"
SOLVER_ERROR = "solver_error"
MAX_ITER_REACHED = "max_iter_reached"
TIME_LIMIT_REACHED = "time_limit_reached"

SOLUTION_PRESENT = [OPTIMAL, OPTIMAL_INACCURATE]

GUROBI_STATUS_MAP = {2: OPTIMAL,
                  3: PRIMAL_INFEASIBLE,
                  5: DUAL_INFEASIBLE,
                  4: PRIMAL_OR_DUAL_INFEASIBLE,
                  6: SOLVER_ERROR,
                  7: MAX_ITER_REACHED,
                  8: SOLVER_ERROR,
                  9: TIME_LIMIT_REACHED,
                  10: SOLVER_ERROR,
                  11: SOLVER_ERROR,
                  12: SOLVER_ERROR,
                  13: OPTIMAL_INACCURATE}

PDQP_STATUS_MAP = {"TERMINATION_REASON_OPTIMAL": OPTIMAL,
                   "TERMINATION_REASON_ITERATION_LIMIT": MAX_ITER_REACHED,
                   "TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT": SOLVER_ERROR,
                   "TERMINATION_REASON_TIME_LIMIT": TIME_LIMIT_REACHED}

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")
OUTPUT_FOLDER = None

def gurobi_solve(solver_type, problem_dir):
    try:
        model = gp.read(problem_dir)
        model.setParam('OutputFlag', 0)
                
        model.setParam("FeasibilityTol", SETTINGS[solver_type]['FeasibilityTol'])
        model.setParam("OptimalityTol", SETTINGS[solver_type]['OptimalityTol'])
        model.setParam("TimeLimit", SETTINGS[solver_type]['TimeLimit'])

        # Update model
        model.update()

        # Solve problem
        model.optimize()
        
        status = GUROBI_STATUS_MAP.get(model.Status, SOLVER_ERROR)
        
        if model.run_time > TIME_LIMIT:
            status = TIME_LIMIT_REACHED
        solution_dict = {'name': [None],
                        'solver': [solver_type],
                        'status': [status],
                        'run_time': [model.run_time],
                        'iter': [model.BarIterCount],
                        'obj_val': [model.objVal],
                        'obj_opt': [None]}
        
        return  pd.DataFrame(solution_dict)
    
    except Exception as e:
        print(f"Gurobi solve error: {e}")
        solution_dict = {'name': [None],
                        'solver': [solver_type],
                        'status': [SOLVER_ERROR],
                        'run_time': [None],
                        'iter': [None],
                        'obj_val': [None],
                        'obj_opt': [None]
                        }
        
        return pd.DataFrame(solution_dict)

def warmup_julia():
    """Warmup Julia JIT compilation with a small problem"""
    jl.seval("""
    # Simple test problem for warmup
    n, m = 2, 1
    P = sparse([1.0 0.0; 0.0 1.0])
    q = [1.0, 1.0]
    A = sparse([1.0 1.0])
    l = [1.0]
    u = [1.0]
    
    variable_lower_bound = fill(-Inf, n)
    variable_upper_bound = fill(Inf, n)
    
    problem_qp = PDQP.QuadraticProgrammingProblem(
        n, m,
        variable_lower_bound, variable_upper_bound,
        isfinite.(variable_lower_bound), isfinite.(variable_upper_bound),
        P, q, 0.0, A, A', l, m
    )
    
    restart_params = PDQP.construct_restart_parameters(
        PDQP.ADAPTIVE_KKT, PDQP.KKT_GREEDY,
        1000, 0.36, 0.2, 0.8, 0.2,
    )
    
    termination_params = PDQP.construct_termination_criteria(
        eps_optimal_absolute = 1e-6,
        eps_optimal_relative = 1e-6,
        time_sec_limit = 10.0,
        iteration_limit = 10,
    )
    
    params = PDQP.PdhgParameters(
        10, true, 1.0, 1.0, true, 0, true, Int32(96),
        termination_params, restart_params,
        PDQP.ConstantStepsizeParams(),
    )
    
    # Run warmup solve (results discarded)
    solver_output = PDQP.optimize(params, problem_qp)
    
    runtime = solver_output.iteration_stats[end].cumulative_time_sec
    println("Runtime (seconds): ", runtime)
    """)
    
    
def pdqp_solve_dir(high_accuracy, solver_type, problem_dir):
    try:
        if high_accuracy:
            jl.tolerance = EPS_HIGH
        else:
            jl.tolerance = EPS_LOW
        jl.time_sec_limit = TIME_LIMIT
        jl.instance_path = problem_dir
        jl.gpu_flag = False
        
        jl.seval("""            
                qp = PDQP.qps_reader_to_standard_form(instance_path)
                
                restart_params = PDQP.construct_restart_parameters(
                    PDQP.ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
                    PDQP.KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
                    1000,                   # restart_frequency_if_fixed
                    0.36,                   # artificial_restart_threshold
                    0.2,                    # sufficient_reduction_for_restart
                    0.8,                    # necessary_reduction_for_restart
                    0.2,                    # primal_weight_update_smoothing
                )
                
                termination_params = PDQP.construct_termination_criteria(
                    # optimality_norm = L2,
                    eps_optimal_absolute = tolerance,
                    eps_optimal_relative = tolerance,
                    time_sec_limit = time_sec_limit,
                    iteration_limit = typemax(Int32),
                    kkt_matrix_pass_limit = Inf,
                )
                
                params = PDQP.PdhgParameters(
                    10,
                    true,
                    1.0,
                    1.0,
                    true,
                    2,
                    true,
                    96,
                    termination_params,
                    restart_params,
                    PDQP.ConstantStepsizeParams(),  
                )
                
                instance_name = replace(basename(instance_path), r"\.(mps|MPS|qps|QPS)(\.gz)?$" => "")
                lower_file_name = lowercase(basename(instance_path))
                solver_output = PDQP.optimize(params, qp)
        """)
        
        # Extract results
        termination_reason = str(jl.seval("string(solver_output.termination_reason)"))
        iteration_count = int(jl.seval("solver_output.iteration_count"))
            
        status = PDQP_STATUS_MAP.get(termination_reason, SOLVER_ERROR)
        
        # Get runtime
        run_time = float(jl.seval("solver_output.iteration_stats[end].cumulative_time_sec"))
        
        # Get objective value
        if jl.seval("length(solver_output.iteration_stats)") > 0:
            obj_val = float(jl.seval("solver_output.iteration_stats[end].convergence_information[1].primal_objective"))
        else:
            obj_val = None
        
        if run_time > TIME_LIMIT:
            status = TIME_LIMIT_REACHED
        
        solution_dict = {
            'name': [None],
            'solver': [solver_type],
            'status': [status],
            'run_time': [run_time],
            'iter': [iteration_count],
            'obj_val': [obj_val if obj_val is not None else None],
            'obj_opt': [None]
        }
        
        return pd.DataFrame(solution_dict)
    
    except Exception as e:
        print(f"PDQP solve error: {e}")
        import traceback
        traceback.print_exc()
        solution_dict = {
            'name': [None],
            'solver': [solver_type],
            'status': [SOLVER_ERROR],
            'run_time': [None],
            'iter': [None],
            'obj_val': [None],
            'obj_opt': [None]
        }
        
        return pd.DataFrame(solution_dict)
    
        
def solve_probs(solver, problem_dir_mps, high_accuracy):
    # Get problems
    lst_probs_mps = [f for f in os.listdir(problem_dir_mps) if f.endswith('.mps')]
    lst_probs_mps.sort()
    lst_probs_paths_mps = [os.path.join(problem_dir_mps, f) for f in lst_probs_mps]
    problem_names_mps = [f[:-4] for f in lst_probs_mps]
    
    all_results = []
    
    if solver.startswith(pdqp):
        for prob_path, name in zip(lst_probs_paths_mps, problem_names_mps):
            result = pdqp_solve_dir(high_accuracy, solver_type=solver, problem_dir=prob_path)
            
            print(f"{name} solved")
            
            result['name'] = name
            result['obj_opt'] = OPT_COST_MAP[name]
            
            all_results.append(result)
    else:
        for prob_path, name in zip(lst_probs_paths_mps, problem_names_mps):
            result = gurobi_solve(solver_type=solver, problem_dir=prob_path)
            print(f"{name} solved")
            
            result['name'] = name
            result['obj_opt'] = OPT_COST_MAP[name]
            
            all_results.append(result)
    
    # Directory of the results
    output_dir = os.path.join(results_dir, OUTPUT_FOLDER)
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, solver)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'results.csv')
    
    # Saving results as a .csv file
    df = pd.concat(all_results)
    df.to_csv(output_path, index=False)
            
    
    return

def main():
    global OUTPUT_FOLDER
    
    parser = argparse.ArgumentParser(description='Performance statistics computation')
    parser.add_argument('--high_accuracy', help='Test with high accuracy (high)', default=False,
                        action='store_true')
    
    args = parser.parse_args()
    high_accuracy = args.high_accuracy
    
    # Warming up julia code
    jl.seval("using Pkg")
    jl.seval('Pkg.develop(path="PDQP.jl")')
    jl.seval('Pkg.add("ArgParse")')
    jl.seval('Pkg.add("JSON3")')
    jl.seval('Pkg.add("GZip")')
    jl.seval('Pkg.add("CUDA")')
    jl.seval("using PDQP")
    jl.seval("using SparseArrays")
    jl.seval("using LinearAlgebra")

    print("Warming up Julia JIT compiler...")
    warmup_julia()
    warmup_julia()
    print("Warmup complete.")
    
    problem_dir_mps = os.path.join(script_dir, "maros")
    problem_dir_mps = os.path.abspath(problem_dir_mps)
    
    OUTPUT_FOLDER = 'Maros'
        
    # Iterate over all solvers
    print(f"problem_dir_mps: {problem_dir_mps}")
    if high_accuracy:
        solvers = [pdqp_high, GUROBI_high]
    else:
        solvers = [pdqp, GUROBI]
    for s in solvers:
        solve_probs(s, problem_dir_mps, high_accuracy)
        
    return
    
    
if __name__ == "__main__":
    main()