# Convex Optimization Practical Q2

# MIMO detection via Convex Optimization

# By: Max Engels, Nathan van Himbergen

# Imports
import numpy as np
import cvxpy as cp
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time

# Open the matlab file and extract variables

data = loadmat('mimo_detection.mat')

# Check the keys in the data dictionary
# print(data.keys())

# HC -> shape (40,40)
hc = data['Hc']

# nrx -> number of receive antennas = 40
nrx = data['Nrx'][0,0]

# ntx -> number of transmit antennas = 40
ntx = data['Ntx'][0,0]

# sc -> shape (40,1)
sc = data['sc']

# yc -> shape (40,1)
yc = data['yc']

# SNR_dB = 20 dB
SNR_dB = data['snrdB'][0,0]

# append complex values to real-valued vectors
def complex_to_real_matrix(Hc):
    H_real = np.zeros((2 * nrx, 2 * ntx))
    H_real[:nrx, :ntx] = Hc.real
    H_real[:nrx, ntx:] = -Hc.imag
    H_real[nrx:, :ntx] = Hc.imag
    H_real[nrx:, ntx:] = Hc.real
    return H_real

H = complex_to_real_matrix(hc)

s_true = np.zeros(2*ntx)
s_true[:ntx] = sc.real.flatten()
s_true[ntx:] = sc.imag.flatten()
y = np.zeros(2*nrx)
y[:nrx] = yc.real.flatten()
y[nrx:] = yc.imag.flatten()

def mimo_detection(H, y, s, n_randomizations=200, verbose=False):
    nrx, ntx = H.shape

    # Construct the quadratic form:
    #   x^T C x = [s; t]^T [ H^T H    -H^T y
    #                           -y^T H    y^T y ] [s; t]
    # which equals ||y - Hs||^2
    Q = H.T @ H          # (n_tx, n_tx)
    b = -H.T @ y         # (n_tx,)
    c = y @ y.T  # scalar

    # Build block matrix C of size (n_tx+1, n_tx+1)
    C = np.zeros((ntx + 1, ntx + 1))
    C[:ntx, :ntx] = Q
    C[:ntx, -1] = b
    C[-1, :ntx] = b
    C[-1, -1] = c

    # SDP variable X \in R^{(n_tx+1) x (n_tx+1)}, symmetric, PSD
    dim = ntx + 1
    X = cp.Variable((dim, dim), symmetric=True)

    # Objective: minimize trace(C X)
    objective = cp.Minimize(cp.trace(C @ X))

    # Constraints:
    constraints = []
    # 1. X must be positive semidefinite
    constraints.append(X >> 0)
    # 2. diag(X) = 1  (corresponds to x_i^2 = 1 -> entries of x are ±1)
    constraints.append(cp.diag(X) == 1)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=verbose)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"SDR solver did not converge. Status: {problem.status}")

    X_star = X.value
    return X_star



def randomization_sdr(X_star, H, y, n_randomizations=200, verbose=False):
    # Eigen-decomposition (for optional rank-1 approximation)
    eigvals, eigvecs = np.linalg.eigh(X_star)

    # We generate L random vectors with covariance X_star:
    #   z ~ N(0, X_star)
    # and project each to {-1,+1}^{dim}, then evaluate cost and keep the best.

    rng = np.random.default_rng(123)

    # To sample from N(0, X_star), we can use Cholesky (if positive definite)
    # or eigen-decomposition for a robust approach.
    eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
    # Build square root of X_star: X_star = U diag(eigvals) U^T
    # sqrt(X_star) = U diag(sqrt(eigvals_clipped)) U^T
    sqrt_diag = np.sqrt(eigvals_clipped)
    sqrt_X = eigvecs @ np.diag(sqrt_diag)
    for _ in range(n_randomizations):
        # Sample z ~ N(0, X_star) by drawing z0 ~ N(0, I) and mapping via sqrt_X
        z0 = rng.normal(0.0, 1.0, size=sqrt_X.shape[0])  # dim = ntx + 1
        z = sqrt_X @ z0

        x_candidate = np.sign(z)  # project to {-1,+1}

        # Recover s from x = [s; t]
        t_cand = x_candidate[-1]
        s_cand = np.sign(x_candidate[:ntx*2] * t_cand)

        residual = np.abs(y - H @ s_cand)
        cost = float(np.dot(residual, residual))

        best_cost = np.inf
        if cost < best_cost:
            best_cost = cost
            best_s = s_cand

    if verbose:
        print(f"SDR cost (after randomization): {best_cost:.4f}")

    return best_s, best_cost

# print("=== MIMO instance ===")
# print(f"H shape: {H.shape}")
# print(f"True symbols s_true: {s_true}")
# print(f"Noise std: {SNR_dB:.4f}")



def projected_gradient_sdr(H, y, s, max_iter=300, step_size=1e-2, verbose=False):
    nrx, ntx = H.shape
    
    # Construct the quadratic form:
    #   x^T C x = [s; t]^T [ H^T H    -H^T y
    #                           -y^T H    y^T y ] [s; t]
    # which equals ||y - Hs||^2
    Q = H.T @ H          # (n_tx, n_tx)
    b = -H.T @ y         # (n_tx,)
    c = y @ y.T  # scalar

    # Build block matrix C of size (n_tx+1, n_tx+1)
    C = np.zeros((ntx + 1, ntx + 1))
    C[:ntx, :ntx] = Q
    C[:ntx, -1] = b
    C[-1, :ntx] = b
    C[-1, -1] = c
    
    d = C.shape[0]

    # Initialize with identity, which is PSD and diag=1
    X = np.eye(d)

    def project_psd_and_diag(X_in):
        # Symmetrize
        X_sym = 0.5 * (X_in + X_in.T)
        # Eigen-decomposition, clamp negative eigenvalues
        eigvals, eigvecs = np.linalg.eigh(X_sym)
        eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
        X_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        # Enforce diag(X) = 1
        np.fill_diagonal(X_psd, 1.0)
        return X_psd

    obj_vals = []
    t0 = time.time()

    for k in range(max_iter):
        # Objective
        obj = float(np.trace(C @ X))
        obj_vals.append(obj)

        if verbose and (k % 20 == 0 or k == max_iter - 1):
            print(f"[PGD] iter {k:4d}, objective = {obj:.6f}")

        # Gradient step: grad f(X) = C
        Y = X - step_size * C

        # Projection step
        X = project_psd_and_diag(Y)

    cpu_time = time.time() - t0
    return X, obj_vals, cpu_time

# ---- Solve SDR-based detector ----
print("Running SDR-based detector...")
# keep track of CPU time
t0 = time.time()

X_star = mimo_detection(H, y, s_true, verbose=False)
s_sdr, sdr_cost = randomization_sdr(
    X_star, H, y, 50, verbose=True
)
#print(f"SDR solution s_sdr:   {s_sdr}")
print(f"SDR cost ||y - Hs||^2: {sdr_cost:.6f}")
print(f"SDR CPU time: {time.time() - t0:.4f} seconds")

# give a small margin of error for numerical inaccuracies
symbol_error = int(np.sum(np.abs(s_sdr - s_true) > 1e-5))
print(f"SDR symbol errors: {symbol_error} out of {2*ntx}")

# ---- Solve SDR via Projected Gradient Descent ----
print("Running PGD-based SDR solver...")
X_pgd, obj_vals, pgd_time = projected_gradient_sdr(
    H, y, s_true, max_iter=10000, step_size=1e-2, verbose=False
)
s_pgd, pgd_cost = randomization_sdr(
    X_pgd, H, y, 50, verbose=True
)
#print(f"PGD SDR solution s_pgd:   {s_pgd}")
print(f"PGD SDR cost ||y - Hs||^2: {pgd_cost:.6f}")
print(f"PGD CPU time: {pgd_time:.4f} seconds")  
symbol_error = int(np.sum(np.abs(s_pgd - s_true) > 1e-5))
print(f"SDR symbol errors: {symbol_error} out of {2*ntx}")
