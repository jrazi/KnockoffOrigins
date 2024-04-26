import numpy as np
from typing import Any, Union

from numpy import ndarray


def compute_gram_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute the Gram matrix of the normalized features.

    Args:
        X: A 2D numpy array (n x p) representing the normalized feature matrix.

    Returns:
        A 2D numpy array (n x n) representing the Gram matrix of X.
    """
    return X.T @ X


def generate_gram_matrix(X: np.ndarray) -> np.ndarray:
    """
    Generate the Gram matrix from the original feature matrix after normalizing features.

    Args:
        X: A 2D numpy array (n x p) representing the original feature matrix.

    Returns:
        A 2D numpy array (n x n) representing the Gram matrix of the normalized features.
    """

    normalized_X = normalize_features(X)
    return compute_gram_matrix(normalized_X)


def normalize_features(X: ndarray) -> ndarray:
    """
    Normalize the columns of the feature matrix so that each feature's L2 norm is 1.

    Args:
        X: A 2D numpy array (n x p) representing the feature matrix.

    Returns:
        A 2D numpy array (n x p) representing the normalized feature matrix.
    """

    gram = compute_gram_matrix(X)

    normalized_X = X / np.sqrt(np.diag(gram))

    return normalized_X
