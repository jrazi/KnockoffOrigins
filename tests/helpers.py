from turtle import pos
import numpy as np
from numpy import ndarray
from typing import Tuple


def generate_matrix(
    dimension: Tuple[int, int], positive_definiteness: bool, symmetrical: bool
) -> ndarray:
    """
    Generate a random matrix with specified properties. Ensures matrices are positive definite,
    symmetric, or explicitly non-positive definite and non-symmetric based on parameters.
    """
    r, c = dimension
    if r != c and positive_definiteness:
        raise ValueError("Positive definite matrices must be square.")

    if positive_definiteness and not symmetrical:
        raise ValueError("Positive definite matrices must be symmetric.")

    # Start with a random matrix
    matrix = np.random.randn(r, c)

    if positive_definiteness:
        matrix = make_positive_definite(matrix)

    elif symmetrical:
        matrix = make_symmetric(matrix)

    elif r == c:
        matrix += np.triu(np.random.rand(r, c), 1)

    return matrix


def make_symmetric(matrix: ndarray) -> ndarray:
    """
    Make a matrix symmetric by averaging with its transpose.
    """
    return (matrix + matrix.T) / 2


def make_positive_definite(matrix: ndarray) -> ndarray:
    """
    Make a matrix positive definite by modifying its eigenvalues.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Only square matrices can be made positive definite.")
    symmetric_matrix = make_symmetric(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix)
    positive_eigenvalues = np.diag(
        np.abs(eigenvalues) + 0.1
    )  # Shift all eigenvalues to be positive
    return eigenvectors @ positive_eigenvalues @ eigenvectors.T
