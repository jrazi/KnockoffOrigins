import numpy as np
from numpy.linalg import cholesky, LinAlgError
from abc import ABC, abstractmethod
from numpy import ndarray


class DecompositionAlgorithm(ABC):
    """
    Abstract class for decomposition algorithms.
    """

    @abstractmethod
    def decompose(self, matrix: ndarray) -> ndarray:
        """
        Method to decompose a matrix. Needs to be implemented by all subclasses.
        """
        pass


class CholeskyDecomposition(DecompositionAlgorithm):
    """
    Cholesky decomposition of a matrix.
    """

    def decompose(self, matrix: ndarray) -> ndarray:
        """
        Perform the Cholesky decomposition on the given matrix.
        Assumes the matrix is symmetric and positive-definite.

        Parameters:
        - matrix (ndarray): The matrix to decompose, which should be symmetric and positive-definite.

        Returns:
        - ndarray: The Cholesky decomposition of the matrix, or raises an error if conditions are not met.

        Raises:
        - ValueError: If the matrix is not square or not symmetric.
        - np.linalg.LinAlgError: If the matrix is not positive-definite.
        """
        if not self._is_square(matrix):
            raise ValueError("Matrix must be square to perform Cholesky decomposition.")
        if not self._is_symmetric(matrix):
            raise ValueError(
                "Matrix must be symmetric to perform Cholesky decomposition."
            )

        try:
            return cholesky(matrix)
        except LinAlgError as e:
            raise ValueError(
                "Matrix must be positive-definite for Cholesky decomposition."
            ) from e

    def _is_square(self, matrix: ndarray) -> bool:
        """
        Check if a matrix is square.

        Parameters:
        - matrix (ndarray): The matrix to check.

        Returns:
        - bool: True if the matrix is square, False otherwise.
        """
        return matrix.shape[0] == matrix.shape[1]

    def _is_symmetric(self, matrix: ndarray) -> bool:
        """
        Check if a matrix is symmetric.

        Parameters:
        - matrix (ndarray): The matrix to check.

        Returns:
        - bool: True if the matrix is symmetric, False otherwise.
        """
        return np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10)


class Decompose:
    """
    Context class that performs matrix decomposition using a specified strategy.
    """

    def __init__(self, decomposition_strategy: DecompositionAlgorithm):
        """
        Initialize with a specific decomposition strategy.
        """
        self.decomposition_strategy = decomposition_strategy

    def decompose(self, matrix: ndarray) -> ndarray:
        """
        Use the strategy to decompose the matrix.
        """
        return self.decomposition_strategy.decompose(matrix)
