import numpy as np
from numpy import ndarray
from scipy.linalg import qr
from knockofforigins.decompose import CholeskyDecomposition
from knockofforigins.gram_matrix import (
    compute_gram_matrix,
    normalize_features,
)


def choose_s_vector(Sigma: ndarray, tau: float = np.inf) -> ndarray:
    """
    Choose the vector s based on the constraints that diag{s} <= 2 * Sigma.

    Parameters:
        Sigma: Covariance matrix of the original features (p, p).
        tau: Optional upper limit for the values of s to ensure positive semidefinite matrix.

    Returns:
        A vector s that satisfies the constraints for knockoff feature generation.
    """
    s = np.zeros(Sigma.shape[0])

    # Ensure diag(s) <= 2 * Sigma
    EPSILON = 1e-3
    for j in range(Sigma.shape[0]):
        s_j = (
            min(2 * Sigma[j, j] - EPSILON, tau)
            if tau is not None
            else 2 * Sigma[j, j] - EPSILON
        )
        s[j] = s_j

    return s


def generate_knockoff_features(
    X: np.ndarray, s: np.ndarray, random_state=42, normalize=True
) -> np.ndarray:
    """
    Construct knockoff features based on the original feature matrix X.

    Args:
        X: Original feature matrix (n x p).
        s: Nonnegative vector controlling knockoff construction (p-dimensional).
        random_state: Optional seed for the random number generator (for testing).

    Returns:
        Knockoff feature matrix (n x p).
    """

    if normalize:
        X = normalize_features(X)

    else:
        X = X

    # Ensure U_tilde is orthogonal and has zero mean adjustment
    U_tilde = np.random.randn(X.shape[0], X.shape[1])
    Q, _ = qr(U_tilde, mode="economic")  # Orthonormal basis
    U_tilde = Q * np.sqrt(X.shape[0])  # Adjust to have proper scaling

    # Compute the knockoff matrix
    I = np.eye(X.shape[1])
    Sigma = compute_gram_matrix(X)

    C = CholeskyDecomposition().decompose(
        2 * np.diag(s) - np.diag(s) @ np.linalg.inv(Sigma) @ np.diag(s)
    )
    X_knockoff = X @ (I - np.linalg.inv(Sigma) @ np.diag(s)) + U_tilde @ C.transpose()

    return X_knockoff
