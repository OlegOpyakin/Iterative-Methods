import numpy as np
from typing import Tuple, List, Optional
import warnings


class IterativeMethodsSolver:
    """
    Solver class for iterative methods to solve linear systems Ax = b.
    
    Args:
        A (np.ndarray): Coefficient matrix (n x n)
        b (np.ndarray): Right-hand side vector (n)
        tolerance (float): Convergence tolerance (default: 1e-10)
        max_iterations (int): Maximum number of iterations (default: 1000)
    """
    
    def __init__(self, A: np.ndarray, 
                 b: np.ndarray, 
                 tolerance: float = 1e-10, 
                 max_iterations: int = 1000):
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square")
        if A.shape[0] != b.shape[0]:
            raise ValueError("Matrix A and vector b dimensions must match")
        
        self.A = A.astype(float)
        self.b = b.astype(float)
        self.n = A.shape[0]
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        
    def _check_convergence(self, x: np.ndarray) -> float:
        """
        Check convergence by computing residual norm ||Ax - b||_2

        Args:
            x (np.ndarray): Vector to calculate residual norm with
            
        Returns
            float: Returns a convergance number for given vector in current system
        """
        residual = self.A @ x - self.b
        return np.linalg.norm(residual)
    
    def _check_diagonal_dominance(self) -> bool:
        """
        Check if matrix has diagonal dominance (important for Jacobi/Gauss-Seidel stability)
        Module of diagonal element must be greater than sum of one element higher and one element lower.

        Returns: 
            bool: Returns True or False, whether matrice is has diagonal stability
        """
        for i in range(self.n):
            diag_val = abs(self.A[i, i])
            off_diag_sum = sum(abs(self.A[i, j]) for j in range(self.n) if i != j)
            if diag_val <= off_diag_sum:
                return False
        return True
    
    def jacobi_method(self, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[float], int]:
        """
        Jacobi Method: x^{k+1}_i = (b_i - sum_{j≠i} A_{ij}x^k_j) / A_{ii}
        
        Mathematical Formula:
            x^{k+1} = D^{-1}(b - (L + U)x^k)
            where A = D + L + U (diagonal, lower, upper triangular parts)

        Args:
            x0 (Optional[np.ndarray]): Vector to start iteration with
        
        Returns:
            np.ndarray : Solution vector
            List[float] : List of residual norms at each iteration
            int : Number of iterations performed
        """
        if not self._check_diagonal_dominance():
            warnings.warn("Matrix lacks diagonal dominance. Jacobi method may not converge.")
        
        # Check for zero diagonal elements
        if np.any(np.diag(self.A) == 0):
            raise ValueError("Zero diagonal element found. Jacobi method cannot be used.")
        
        x = np.zeros(self.n) if x0 is None else x0.copy()
        residual_norms = []
        
        D_inv = np.diag(1.0 / np.diag(self.A))  # Inverse of diagonal matrix
        R = self.A - np.diag(np.diag(self.A))  # A - D (off-diagonal part)
        
        for iteration in range(self.max_iterations):
            residual_norm = self._check_convergence(x)
            residual_norms.append(residual_norm)
            
            if residual_norm < self.tolerance:
                return x, residual_norms, iteration
            
            x = D_inv @ (self.b - R @ x)
        
        warnings.warn(f"Max iterations ({self.max_iterations}) reached without convergence")
        return x, residual_norms, self.max_iterations
    
    def gauss_seidel_method(self, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[float], int]:
        """
        Gauss-Seidel Method: x^{k+1}_i = (b_i - sum_{j<i} A_{ij}x^{k+1}_j - sum_{j>i} A_{ij}x^k_j) / A_{ii}
        
        Mathematical Formula:
            (D + L)x^{k+1} = b - Ux^k
            where A = D + L + U
        
        Args: 
            x0 (Optional[np.ndarray]): Vector to start iteration with

        Returns:
            np.ndarray : Solution vector
            List[float] : List of residual norms at each iteration
            int : Number of iterations performed
        """
        # Check for zero diagonal elements
        if np.any(np.diag(self.A) == 0):
            raise ValueError("Zero diagonal element found. Gauss-Seidel method cannot be used.")
        
        x = np.zeros(self.n) if x0 is None else x0.copy()
        residual_norms = []
        
        for iteration in range(self.max_iterations):
            residual_norm = self._check_convergence(x)
            residual_norms.append(residual_norm)
            
            if residual_norm < self.tolerance:
                return x, residual_norms, iteration
            
            x_new = x.copy()
            for i in range(self.n):
                sum1 = sum(self.A[i, j] * x_new[j] for j in range(i))
                sum2 = sum(self.A[i, j] * x[j] for j in range(i + 1, self.n))
                x_new[i] = (self.b[i] - sum1 - sum2) / self.A[i, i]
            x = x_new
        
        warnings.warn(f"Max iterations ({self.max_iterations}) reached without convergence")
        return x, residual_norms, self.max_iterations
    
    def simple_iteration_method(self, tau: float = 0.001, 
                              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[float], int]:
        """
        Simple Iteration Method: x^{k+1} = x^k - τ(Ax^k - b)
        
        Mathematical Formula:
            x^{k+1} = x^k - τr^k, where r^k = Ax^k - b is the residual
            
        Args:
            tau (float): Step size parameter     
            x0 (Optional[np.ndarray]): Vector to start iteration with  
        
        Returns:   
            np.ndarray : Solution vector
            List[float] : List of residual norms at each iteration
            int : Number of iterations performed
        """
        #warnings.warn(f"Max iterations ({self.max_iterations}) reached without convergence")

        x = np.zeros(self.n) if x0 is None else x0.copy()
        residual_norms = []
        
        for iteration in range(self.max_iterations):
            residual_norm = self._check_convergence(x)
            residual_norms.append(residual_norm)
            
            if residual_norm < self.tolerance:
                return x, residual_norms, iteration
            
            residual = self.A @ x - self.b
            x = x - tau * residual
        
        warnings.warn(f"Max iterations ({self.max_iterations}) reached without convergence")
        return x, residual_norms, self.max_iterations
    
    def steepest_gradient_descent(self, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[float], int]:
        """
        Steepest Gradient Descent Method: x^{k+1} = x^k - τ_k r^k
        where τ_k = ||r^k||^2 / (r^k)^T A r^k (optimal step size)
        
        Mathematical Formula:
            For solving Ax = b, minimize f(x) = 0.5 * x^T A x - b^T x
            Gradient: ∇f(x) = Ax - b = r (residual)
            Optimal step size: τ_k = r^T r / r^T A r
        
        Args:
            x0 (Optional[np.ndarray]): Vector to start iteration with

        Returns:
            np.ndarray : Solution vector
            List[float] : List of residual norms at each iteration
            int : Number of iterations performed
        """
        x = np.zeros(self.n) if x0 is None else x0.copy()
        residual_norms = []
        
        for iteration in range(self.max_iterations):
            residual = self.A @ x - self.b
            residual_norm = np.linalg.norm(residual)
            residual_norms.append(residual_norm)
            
            if residual_norm < self.tolerance:
                return x, residual_norms, iteration
            
            Ar = self.A @ residual
            tau = np.dot(residual, residual) / np.dot(residual, Ar)
            x = x - tau * residual
        
        warnings.warn(f"Max iterations ({self.max_iterations}) reached without convergence")
        return x, residual_norms, self.max_iterations
    
    def successive_over_relaxation(self, omega: Optional[float] = None, 
                                 x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[float], int]:
        """
        Successive Over-Relaxation (SOR) Method
        
        Mathematical Formula:
            x^{k+1}_i = (1-ω)x^k_i + ω/A_{ii}[b_i - sum_{j<i} A_{ij}x^{k+1}_j - sum_{j>i} A_{ij}x^k_j]
        
        Args:
            omega (Optional[np.ndarray]): relaxation parameter (if None, computed automatically)
            x0 (Optional[np.ndarray]): Vector to start iteration with

        Returns:
            np.ndarray : Solution vector
            List[float] : List of residual norms at each iteration
            int : Number of iterations performed
        """
        if np.any(np.diag(self.A) == 0):
            raise ValueError("Zero diagonal element found. SOR method cannot be used.")
        
        if omega is None:
            # Estimate optimal omega (simplified approach)
            # For better results, one would compute spectral radius
            omega = 1.0  # Default to Gauss-Seidel
        
        x = np.zeros(self.n) if x0 is None else x0.copy()
        residual_norms = []
        
        for iteration in range(self.max_iterations):
            residual_norm = self._check_convergence(x)
            residual_norms.append(residual_norm)
            
            if residual_norm < self.tolerance:
                return x, residual_norms, iteration
            
            for i in range(self.n):
                sum1 = sum(self.A[i, j] * x[j] for j in range(i))
                sum2 = sum(self.A[i, j] * x[j] for j in range(i + 1, self.n))
                
                x_old = x[i]
                x_new = (self.b[i] - sum1 - sum2) / self.A[i, i]
                x[i] = (1 - omega) * x_old + omega * x_new
        
        warnings.warn(f"Max iterations ({self.max_iterations}) reached without convergence")
        return x, residual_norms, self.max_iterations
    
    def symmetric_gauss_seidel(self, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[float], int]:
        """
        Symmetric Gauss-Seidel Method (forward and backward sweeps)
        
        Mathematical Formula:
            Forward sweep: (D + L)x^{k+1/2} = b - Ux^k
            Backward sweep: (D + U)x^{k+1} = b - Lx^{k+1/2}
        
        Args:
            x0 (Optional[np.ndarray]): Vector to start iteration with
            
        Returns:
            np.ndarray : Solution vector
            List[float] : List of residual norms at each iteration
            int : Number of iterations performed
        """
        if np.any(np.diag(self.A) == 0):
            raise ValueError("Zero diagonal element found. Symmetric Gauss-Seidel method cannot be used.")
        
        x = np.zeros(self.n) if x0 is None else x0.copy()
        residual_norms = []
        
        for iteration in range(self.max_iterations):
            residual_norm = self._check_convergence(x)
            residual_norms.append(residual_norm)
            
            if residual_norm < self.tolerance:
                return x, residual_norms, iteration
            
            # Forward sweep (like Gauss-Seidel)
            for i in range(self.n):
                sum1 = sum(self.A[i, j] * x[j] for j in range(i))
                sum2 = sum(self.A[i, j] * x[j] for j in range(i + 1, self.n))
                x[i] = (self.b[i] - sum1 - sum2) / self.A[i, i]
            
            # Backward sweep
            for i in range(self.n - 1, -1, -1):
                sum1 = sum(self.A[i, j] * x[j] for j in range(i))
                sum2 = sum(self.A[i, j] * x[j] for j in range(i + 1, self.n))
                x[i] = (self.b[i] - sum1 - sum2) / self.A[i, i]
        
        warnings.warn(f"Max iterations ({self.max_iterations}) reached without convergence")
        return x, residual_norms, self.max_iterations
    

def create_test_system(n: int = 100, condition_number: float = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a test linear system Ax = b with known solution.
    
    Args:
        n (int): size of the system
        condition_number (float): condition number of matrix A
    
    Returns:
        A : coefficient matrix
        b : right-hand side vector
        x_true : true solution
    """
    # Generate random symmetric positive definite matrix
    np.random.seed(42)  # For reproducibility
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Create eigenvalues with specified condition number
    eigenvals = np.linspace(1, condition_number, n)
    A = Q @ np.diag(eigenvals) @ Q.T
    
    # Create known solution and corresponding b
    x_true = np.random.randn(n)
    b = A @ x_true
    
    return A, b, x_true