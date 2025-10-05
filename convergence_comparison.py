import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
from typing import Dict, List, Tuple, Any
import warnings
from iterative_methods import IterativeMethodsSolver, create_test_system

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def compare_methods(A: np.ndarray, b: np.ndarray, x_true: np.ndarray = None,
                   tolerance: float = 1e-12, max_iterations: int = 500) -> Dict[str, Any]:
    """
    Compare different iterative methods on the same linear system.
    
    Args:
        A (np.ndarray): coefficient matrix
        b (np.ndarray): right-hand side vector
        x_true (np.ndarray): true solution (optional, for error analysis)
        tolerance (float): convergence tolerance
        max_iterations (int): maximum number of iterations
    
    Returns:
        Dict[str, Any] : Dictionary containing results for all methods
    """
    solver = IterativeMethodsSolver(A, b, tolerance, max_iterations)
    results = {}
    
    methods = [
        ('Jacobi', solver.jacobi_method, {}),
        ('Gauss-Seidel', solver.gauss_seidel_method, {}),
        ('Simple Iteration (τ=0.001)', solver.simple_iteration_method, {'tau': 0.001}),
        ('Simple Iteration (τ=0.01)', solver.simple_iteration_method, {'tau': 0.01}),
        ('Steepest Gradient Descent', solver.steepest_gradient_descent, {}),
        ('SOR (ω=1.0)', solver.successive_over_relaxation, {'omega': 1.0}),
        ('SOR (ω=1.2)', solver.successive_over_relaxation, {'omega': 1.2}),
        ('Symmetric Gauss-Seidel', solver.symmetric_gauss_seidel, {})
    ]
    
    for method_name, method_func, kwargs in methods:
        try:
            print(f"Running {method_name}...")
            start_time = time.time()
            solver = IterativeMethodsSolver(A, b, tolerance, max_iterations)
            
            x_sol, residual_norms, iterations = method_func(**kwargs)
            end_time = time.time()
            
            # Calculate error if true solution is known
            error_norms = []
            if x_true is not None:
                error_norms = [np.linalg.norm(x_sol - x_true)]
            
            results[method_name] = {
                'solution': x_sol,
                'residual_norms': residual_norms,
                'error_norms': error_norms,
                'iterations': iterations,
                'converged': residual_norms[-1] < tolerance,
                'final_residual': residual_norms[-1],
                'computation_time': end_time - start_time
            }
            
            print(f"  Converged: {results[method_name]['converged']}")
            print(f"  Iterations: {iterations}")
            print(f"  Final residual: {residual_norms[-1]:.2e}")
            print(f"  Time: {end_time - start_time:.4f}s")
            
        except Exception as e:
            print(f"  Failed: {str(e)}")
            continue
    
    return results


def plot_convergence_comparison(results: Dict[str, Any], title: str = "Convergence Comparison"):
    """
    Plot residual norm convergence for all methods.
    """
    plt.figure(figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    for i, (method_name, data) in enumerate(results.items()):
        if not data['residual_norms']:
            continue
        
        residual_norms = data['residual_norms']
        iterations = range(len(residual_norms))
        
        plt.semilogy(iterations, residual_norms, 
                    color=colors[i], 
                    linestyle=line_styles[i % len(line_styles)],
                    linewidth=2, 
                    marker='o' if len(residual_norms) < 50 else None,
                    markersize=4,
                    label=f"{method_name} ({data['iterations']} iter)")
    
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm ||Ax - b||₂')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_convergence_rate_analysis(results: Dict[str, Any]):
    """
    Analyze and plot convergence rates.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Convergence rate (ratio of consecutive residuals)
    for method_name, data in results.items():
        residual_norms = data['residual_norms']
        if len(residual_norms) < 10:
            continue
            
        # Calculate convergence rate: r_{k+1} / r_k
        rates = []
        for i in range(1, min(50, len(residual_norms))):  # Limit to first 50 iterations
            if residual_norms[i-1] > 1e-15:  # Avoid division by very small numbers
                rate = residual_norms[i] / residual_norms[i-1]
                rates.append(rate)
        
        if rates:
            iterations = range(1, len(rates) + 1)
            ax1.plot(iterations, rates, 'o-', markersize=4, linewidth=1, 
                    label=f"{method_name} (avg: {np.mean(rates):.3f})")
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Convergence Rate (||r^{k+1}|| / ||r^k||)')
    ax1.set_title('Convergence Rate Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.2)
    
    # Plot 2: Performance comparison (iterations vs time)
    methods = list(results.keys())
    iterations = [results[m]['iterations'] for m in methods if results[m]['converged']]
    times = [results[m]['computation_time'] for m in methods if results[m]['converged']]
    converged_methods = [m for m in methods if results[m]['converged']]
    
    if iterations and times:
        scatter = ax2.scatter(iterations, times, s=100, alpha=0.7, c=range(len(iterations)), cmap='viridis')
        
        for i, method in enumerate(converged_methods):
            ax2.annotate(method, (iterations[i], times[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Iterations to Convergence')
        ax2.set_ylabel('Computation Time (seconds)')
        ax2.set_title('Performance Analysis')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_method_comparison_table(results: Dict[str, Any]):
    """Create a comparison table of method performance."""
    # Prepare data for table
    methods = []
    iterations_list = []
    final_residuals = []
    times = []
    converged_list = []
    
    for method_name, data in results.items():
        methods.append(method_name)
        iterations_list.append(data['iterations'])
        final_residuals.append(f"{data['final_residual']:.2e}")
        times.append(f"{data['computation_time']:.4f}")
        converged_list.append("✓" if data['converged'] else "✗")
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for i in range(len(methods)):
        table_data.append([
            methods[i],
            converged_list[i], 
            iterations_list[i],
            final_residuals[i],
            times[i]
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Method', 'Converged', 'Iterations', 'Final Residual', 'Time (s)'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.3, 0.1, 0.15, 0.2, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color code the table
    for i in range(len(methods)):
        if converged_list[i] == "✓":
            table[(i+1, 1)].set_facecolor('#90EE90')  # Light green for converged
        else:
            table[(i+1, 1)].set_facecolor('#FFB6C1')  # Light red for not converged
    
    plt.title('Iterative Methods Comparison Table', fontsize=14, fontweight='bold', pad=20)
    plt.show()


def analyze_matrix_properties(A: np.ndarray):
    """Analyze and display matrix properties."""
    print("\n" + "="*60)
    print("MATRIX PROPERTIES ANALYSIS")
    print("="*60)
    
    # Basic properties
    print(f"Matrix size: {A.shape[0]} × {A.shape[1]}")
    print(f"Condition number: {np.linalg.cond(A):.2e}")
    
    # Eigenvalue analysis
    eigenvals = np.linalg.eigvals(A)
    print(f"Largest eigenvalue: {np.max(eigenvals):.4f}")
    print(f"Smallest eigenvalue: {np.min(eigenvals):.4f}")
    print(f"Spectral radius: {np.max(np.abs(eigenvals)):.4f}")
    
    # Matrix type analysis
    is_symmetric = np.allclose(A, A.T)
    is_positive_definite = is_symmetric and np.all(eigenvals > 0)
    
    print(f"Symmetric: {is_symmetric}")
    print(f"Positive definite: {is_positive_definite}")
    
    # Diagonal dominance check
    diag_dominant = True
    for i in range(A.shape[0]):
        diag_val = abs(A[i, i])
        off_diag_sum = sum(abs(A[i, j]) for j in range(A.shape[1]) if i != j)
        if diag_val <= off_diag_sum:
            diag_dominant = False
            break
    
    print(f"Diagonally dominant: {diag_dominant}")
    print("="*60)

def main():
    """Main function to run the convergence comparison analysis."""
    print("Iterative Methods Convergence Comparison")
    print("="*50)
    
    # Test case 1: Well-conditioned symmetric positive definite matrix
    print("\nTest Case 1: Well-conditioned SPD matrix (condition number ≈ 10)")
    A1, b1, x_true1 = create_test_system(n=50, condition_number=10)
    analyze_matrix_properties(A1)
    
    results1 = compare_methods(A1, b1, x_true1, tolerance=1e-12, max_iterations=200)
    
    plot_convergence_comparison(results1, "Convergence: Well-conditioned SPD Matrix")
    plot_convergence_rate_analysis(results1)
    plot_method_comparison_table(results1)

if __name__ == "__main__":
    main()