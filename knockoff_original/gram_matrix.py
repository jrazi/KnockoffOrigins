import numpy as np
from numpy import ndarray


def normalize_features(X: ndarray) -> ndarray:
    """
    Normalize the columns of the feature matrix so that each feature's L2 norm is 1.

    Parameters:
    - X (ndarray): The original feature matrix with shape (n, p).

    Returns:
    - ndarray: The normalized feature matrix.
    """
    norms = np.linalg.norm(X, axis=0)
    normalized_X = X / norms
    return normalized_X


def compute_gram_matrix(X: ndarray) -> ndarray:
    """
    Compute the Gram matrix of the normalized features.

    Parameters:
    - X (ndarray): The normalized feature matrix with shape (n, p).

    Returns:
    - ndarray: The Gram matrix, which is X^T * X.
    """
    return np.dot(X.T, X)


def generate_gram_matrix(X: ndarray) -> ndarray:
    """
    Generate the Gram matrix from the original feature matrix after normalizing features.

    Parameters:
    - X (ndarray): The original feature matrix with shape (n, p).

    Returns:
    - ndarray: The Gram matrix of the normalized features.
    """
    normalized_X = normalize_features(X)
    gram_matrix = compute_gram_matrix(normalized_X)
    return gram_matrix
