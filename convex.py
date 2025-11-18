# Convex Optimization Practical Q2

# MIMO detection via Convex Optimization

# By: Max Engels, Nathan van Himbergen

# Imports
import numpy as np
import cvxpy as cp
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Open the matlab file and extract variables

data = loadmat('mimo_detection.mat')

# Check the keys in the data dictionary
print(data.keys())

# HC -> shape (40,40)
hc = data['Hc']

# nrx -> number of receive antennas = 40
nrx = data['Nrx'][0,0]

# ntx -> number of transmit antennas = 40
ntx = data['Ntx'][0,0]

# sc -> shape (40,1)
sc = data['sc']
print(sc)
# yc -> shape (40,1)
yc = data['yc']
print(yc)

# SNR_dB = 20 dB
SNR_dB = data['snrdB'][0,0]

# append complex values to real-valued vectors
def complex_to_real_matrix(Hc):
    n_rx, n_tx = Hc.shape
    H_real = np.zeros((2 * n_rx, 2 * n_tx))
    H_real[:n_rx, :n_tx] = Hc.real
    H_real[:n_rx, n_tx:] = -Hc.imag
    H_real[n_rx:, :n_tx] = Hc.imag
    H_real[n_rx:, n_tx:] = Hc.real
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

    # Ensure X_star is symmetric (small numerical asymmetries may exist)
    # X_star = 0.5 * (X_star + X_star.T)

    # Eigen-decomposition (for optional rank-1 approximation)
    eigvals, eigvecs = np.linalg.eigh(X_star)

    # ---- Rank-1 approximate solution (deterministic) ----
    # Take largest eigenvalue/vector
    idx_max = np.argmax(eigvals)
    v = eigvecs[:, idx_max]
    x_rank1 = np.sign(v)  # project to {-1,+1}

    # Recover s from x = [s; t] via s = sign( s_part * t )
    t_rank1 = x_rank1[-1]
    s_rank1 = np.sign(x_rank1[:ntx] * t_rank1)

    residual_rank1 = y - H @ s_rank1
    cost_rank1 = float(np.dot(residual_rank1, residual_rank1))

    # ---- Gaussian randomization (stochastic) ----
    # We generate L random vectors with covariance X_star:
    #   z ~ N(0, X_star)
    # and project each to {-1,+1}^{dim}, then evaluate cost and keep the best.
    best_s = s_rank1.copy()
    best_cost = cost_rank1

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
        z0 = rng.normal(0.0, 1.0, size=dim)
        z = sqrt_X @ z0

        x_candidate = np.sign(z)  # project to {-1,+1}

        # Recover s from x = [s; t]
        t_cand = x_candidate[-1]
        s_cand = np.sign(x_candidate[:ntx] * t_cand)

        residual = y - H @ s_cand
        cost = float(np.dot(residual, residual))

        if cost < best_cost:
            best_cost = cost
            best_s = s_cand

    if verbose:
        print(f"SDR cost (rank-1 approximation): {cost_rank1:.4f}")
        print(f"SDR cost (after randomization): {best_cost:.4f}")

    return best_s, best_cost


# ---- Experiment parameters ----
seed = 42
n_randomizations = 200


print("=== MIMO instance ===")
print(f"H shape: {H.shape}")
print(f"True symbols s_true: {s_true}")
print(f"Noise std: {SNR_dB:.4f}")

# ---- Solve SDR-based detector ----
print("Running SDR-based detector...")
s_sdr, sdr_cost = mimo_detection(
    H, y, s_true, n_randomizations=n_randomizations, verbose=True
)
print(f"SDR solution s_sdr:   {s_sdr}")
print(f"SDR cost ||y - Hs||^2: {sdr_cost:.6f}")

# give a small margin of error for numerical inaccuracies
symbol_error = int(np.sum(np.abs(s_sdr - s_true) > 1e-5))
print(f"SDR symbol errors: {symbol_error} out of {2*ntx}")
print(np.sum(s_true - s_sdr))


