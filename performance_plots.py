import os
import pandas as pd
import numpy as np
import time
import traceback
import argparse
import sys

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

MAX_TIMING = 1000.

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

# Solvers
GUROBI = 'GUROBI'
GUROBI_high = GUROBI + "_high"
pdqp = 'pdqp'
pdqp_high = pdqp + '_high'

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")
OUTPUT_FOLDER = None

def plot_performance_profiles(problems, solvers):
    """
    Plot performance profiles in matplotlib for specified problems and solvers
    """
    solvers = solvers.copy()

    df = pd.read_csv('./results/%s/performance_profiles.csv' % problems)
    plt.figure(0)
    for solver in solvers:
        plt.plot(df["tau"], df[solver], label=solver)
    plt.xlim(1., 10000.)
    plt.ylim(0., 1.)
    plt.xlabel(r'Performance ratio $\tau$')
    plt.ylabel('Ratio of problems solved')
    plt.xscale('log')
    plt.legend()
    plt.grid()
    plt.show(block=False)
    results_file = './results/%s/%s.pdf' % (problems, problems)
    print("Saving plots to %s" % results_file)
    plt.savefig(results_file)


def get_cumulative_data(solvers, problems, output_folder):
    for solver in solvers:

        # Path where solver results are stored
        path = os.path.join('.', 'results', output_folder, solver)

        # Initialize cumulative results
        results = []
        for problem in problems:
            file_name = os.path.join(path, problem, 'full.csv')
            results.append(pd.read_csv(file_name))

        # Create cumulative dataframe
        print(f"results['name']: {results['name']}")
        df = pd.concat(results)

        # Store dataframe into results
        solver_file_name = os.path.join(path, 'results.csv')
        df.to_csv(solver_file_name, index=False)


def compute_performance_profiles(solvers, problems_type):
    t = {}
    status = {}

    # Get time and status
    for solver in solvers:
        path = os.path.join('.', 'results', problems_type,
                            solver, 'results.csv')
        df = pd.read_csv(path)

        # Get total number of problems
        n_problems = len(df)

        t[solver] = df['run_time'].values
        status[solver] = df['status'].values

        # Set maximum time for solvers that did not succeed
        for idx in range(n_problems):
            if status[solver][idx] not in SOLUTION_PRESENT:
                t[solver][idx] = MAX_TIMING

    r = {}  # Dictionary of relative times for each solver/problem
    for s in solvers:
        r[s] = np.zeros(n_problems)

    # Iterate over all problems to find best timing between solvers
    for p in range(n_problems):

        # Get minimum time
        min_time = np.min([t[s][p] for s in solvers])

        # Normalize t for minimum time
        for s in solvers:
            r[s][p] = t[s][p]/min_time

    # Compute curve for all solvers
    n_tau = 1000
    tau_vec = np.logspace(0, 4, n_tau)
    rho = {'tau': tau_vec}  # Dictionary of all the curves

    for s in solvers:
        rho[s] = np.zeros(n_tau)
        for tau_idx in range(n_tau):
            count_problems = 0  # Count number of problems with t[p, s] <= tau
            for p in range(n_problems):
                if r[s][p] <= tau_vec[tau_idx]:
                    count_problems += 1
            rho[s][tau_idx] = count_problems / n_problems

    # Store final pandas dataframe
    df_performance_profiles = pd.DataFrame(rho)
    performance_profiles_file = os.path.join('.', 'results',
                                             problems_type,
                                             'performance_profiles.csv')
    df_performance_profiles.to_csv(performance_profiles_file, index=False)

    # Plot performance profiles
    # import matplotlib.pylab as plt
    # for s in solvers:
    #     plt.plot(tau_vec, rho[s], label=s)
    # plt.legend(loc='best')
    # plt.ylabel(r'$\rho_{s}$')
    # plt.xlabel(r'$\tau$')
    # plt.grid()
    # plt.xscale('log')
    # plt.show(block=False)

def geom_mean(t, shift=10.):
    """Compute the shifted geometric mean using formula from
    http://plato.asu.edu/ftp/shgeom.html

    NB. Use logarithms to avoid numeric overflows
    """
    return np.exp(np.sum(np.log(np.maximum(1, t + shift))/len(t))) - shift


def compute_shifted_geometric_means(solvers, problems_type):
    t = {}
    status = {}
    g_mean = {}

    solvers = solvers.copy()

    # Get time and status
    for solver in solvers:
        path = os.path.join('.', 'results', problems_type,
                            solver, 'results.csv')
        df = pd.read_csv(path)

        # Get total number of problems
        n_problems = len(df)

        # NB. Normalize to avoid overflow. They get normalized back anyway.
        t[solver] = df['run_time'].values
        status[solver] = df['status'].values

        # Set maximum time for solvers that did not succeed
        for idx in range(n_problems):
            if status[solver][idx] not in SOLUTION_PRESENT:
                t[solver][idx] = MAX_TIMING

        g_mean[solver] = geom_mean(t[solver])

    # Normalize geometric means by best solver
    best_g_mean = np.min([g_mean[s] for s in solvers])
    for s in solvers:
        g_mean[s] /= best_g_mean

    # Store final pandas dataframe
    df_g_mean = pd.Series(g_mean)
    g_mean_file = os.path.join('.', 'results',
                               problems_type,
                               'geom_mean.csv')
    df_g_mean.to_frame().transpose().to_csv(g_mean_file, index=False)


def compute_failure_rates(solvers, problems_type):
    """
    Compute and show failure rates
    """
    failure_rates = {}

    solvers = solvers.copy()

    # Check if results file already exists
    failure_rates_file = os.path.join(".", "results", problems_type, "failure_rates.csv")
    for solver in solvers:
        results_file = os.path.join('.', 'results', problems_type,
                                    solver, 'results.csv')
        df = pd.read_csv(results_file)

        n_problems = len(df)

        failed_statuses = np.logical_and(*[df['status'].values != s
                                           for s in SOLUTION_PRESENT])
        n_failed_problems = np.sum(failed_statuses)
        failure_rates[solver] = 100 * (n_failed_problems / n_problems)

    # Write csv file
    df_failure_rates = pd.Series(failure_rates)
    df_failure_rates.to_frame().transpose().to_csv(failure_rates_file, index=False)

def compute_stats_info(solvers, benchmark_type,
                       problems=None,
                       high_accuracy=False,
                       performance_profiles=True):

    print(f"solvers: {solvers}")
    print(f"problems: {problems}")
    if problems is not None:
        # Collect cumulative data for each solver
        # If there are multiple problems defined
        get_cumulative_data(solvers, problems, benchmark_type)

    # Compute failure rates
    compute_failure_rates(solvers, benchmark_type)

    # Compute performance profiles
    compute_performance_profiles(solvers, benchmark_type)

    # Compute performance profiles
    compute_shifted_geometric_means(solvers, benchmark_type)

    # Plot performance profiles
    if performance_profiles:
        plot_performance_profiles(benchmark_type, solvers)


def main():
    global OUTPUT_FOLDER
    
    parser = argparse.ArgumentParser(description='Performance statistics computation')
    parser.add_argument('--high_accuracy', help='Test with high accuracy', default=False,
                        action='store_true')
    parser.add_argument('--problem_set', help='Which benchmark to use (Maros, NetLib)', default='Maros')
    
    args = parser.parse_args()
    high_accuracy = args.high_accuracy
    problem_set = args.problem_set
    
    if (problem_set != "Maros") and (problem_set != 'NetLib'):
        print(f"problme_set must be either Maros or NetLib")
        sys.exit(1)
    
    solvers = [GUROBI, pdqp]
    
    OUTPUT_FOLDER = problem_set
        
    compute_stats_info(solvers, OUTPUT_FOLDER, high_accuracy=high_accuracy)
    return

if __name__ == "__main__":
    main()