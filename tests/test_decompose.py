import numpy as np
import pytest
from knockoff_original.decompose import CholeskyDecomposition
from tests.helpers import generate_matrix


def test_cholesky_decomposition_valid_decomposition():
    """Test if the Cholesky decomposition reconstructs the original matrix."""
    # Generate a positive definite, symmetric matrix
    matrix = generate_matrix((5, 5), positive_definiteness=True, symmetrical=True)
    cholesky = CholeskyDecomposition()
    L = cholesky.decompose(matrix)

    # Reconstruct the original matrix from L
    reconstructed_matrix = L @ L.T
    np.testing.assert_array_almost_equal(
        matrix,
        reconstructed_matrix,
        decimal=6,
        err_msg="Reconstructed matrix does not match original",
    )


def test_cholesky_decomposition_lower_triangular():
    """Test if the output matrix L is lower triangular with positive diagonals."""
    matrix = generate_matrix((5, 5), positive_definiteness=True, symmetrical=True)
    cholesky = CholeskyDecomposition()
    L = cholesky.decompose(matrix)

    assert np.allclose(L, np.tril(L)), "Matrix L is not lower triangular"
    assert np.all(np.diag(L) > 0), "Diagonal elements of L are not all positive"


def test_cholesky_decomposition_non_square_matrix():
    """Test handling of non-square matrices."""
    matrix = generate_matrix((3, 5), positive_definiteness=False, symmetrical=False)
    cholesky = CholeskyDecomposition()
    with pytest.raises(ValueError) as e:
        cholesky.decompose(matrix)
    assert "Matrix must be square" in str(e.value)


def test_cholesky_decomposition_non_symmetric_matrix():
    """Test handling of non-symmetric matrices."""
    matrix = generate_matrix((5, 5), positive_definiteness=False, symmetrical=False)
    cholesky = CholeskyDecomposition()
    with pytest.raises(ValueError) as e:
        cholesky.decompose(matrix)
    assert "Matrix must be symmetric" in str(e.value)


def test_cholesky_decomposition_non_positive_definite_matrix():
    """Test handling of matrices that are not positive definite."""
    matrix = generate_matrix((4, 4), positive_definiteness=False, symmetrical=True)

    cholesky = CholeskyDecomposition()
    with pytest.raises(ValueError) as e:
        cholesky.decompose(matrix)
    assert "Matrix must be positive-definite for Cholesky decomposition" in str(e.value)


def test_cholesky_decomposition_non_symmetric_matrix_manual():
    """Test handling of non-symmetric matrices."""
    # Manually create a non-symmetric matrix
    matrix = np.array([[1, 2], [3, 4]])  # Clearly non-symmetric
    cholesky = CholeskyDecomposition()
    with pytest.raises(ValueError) as e:
        cholesky.decompose(matrix)
    assert "Matrix must be symmetric" in str(e.value)
