import numpy as np
import pytest
from knockoff_original.gram_matrix import (
    normalize_features,
    compute_gram_matrix,
    generate_gram_matrix,
)
from knockoff_original.data_gen import SyntheticDataGenerator, GWASDataGenerator
import numpy as np
import pytest


@pytest.fixture
def synthetic_data():
    generator = SyntheticDataGenerator(n=100, p=20, noise_variance=1.0)
    X, _ = generator.generate_data()
    return X


@pytest.fixture
def gwas_data():
    base_generator = SyntheticDataGenerator(n=100, p=20, noise_variance=1.0)
    gwas_generator = GWASDataGenerator(base_generator)
    X_gwas, _ = gwas_generator.generate_data()
    return X_gwas


def test_normalize_features_unit_norm(synthetic_data):
    normalized_X = normalize_features(synthetic_data)
    norms = np.linalg.norm(normalized_X, axis=0)
    np.testing.assert_array_almost_equal(norms, np.ones(norms.size), decimal=6)


def test_gram_matrix_symmetry(synthetic_data):
    normalized_X = normalize_features(synthetic_data)
    gram_matrix = compute_gram_matrix(normalized_X)
    assert np.allclose(gram_matrix, gram_matrix.T), "Gram matrix is not symmetric"


def test_gram_matrix_diagonal_elements(synthetic_data):
    normalized_X = normalize_features(synthetic_data)
    gram_matrix = compute_gram_matrix(normalized_X)
    diagonal_elements = np.diag(gram_matrix)
    # Check if diagonal elements are close to 1, as each column vector should have unit length
    np.testing.assert_array_almost_equal(
        diagonal_elements, np.ones(diagonal_elements.size), decimal=6
    )


def test_generate_gram_matrix_integration(synthetic_data):
    gram_matrix = generate_gram_matrix(synthetic_data)
    # Ensure matrix is symmetric and check diagonal as a basic integration test
    assert np.allclose(gram_matrix, gram_matrix.T), "Gram matrix is not symmetric"
    diagonal_elements = np.diag(gram_matrix)
    np.testing.assert_array_almost_equal(
        diagonal_elements, np.ones(diagonal_elements.size), decimal=6
    )


def test_gram_matrix_properties_with_gwas_data(gwas_data):
    normalized_X = normalize_features(gwas_data)
    gram_matrix = compute_gram_matrix(normalized_X)
    # Check symmetry and diagonal elements specifically for GWAS data
    assert np.allclose(
        gram_matrix, gram_matrix.T
    ), "Gram matrix from GWAS data is not symmetric"
    diagonal_elements = np.diag(gram_matrix)
    np.testing.assert_array_almost_equal(
        diagonal_elements, np.ones(diagonal_elements.size), decimal=6
    )
