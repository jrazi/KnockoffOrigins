import numpy as np
import pytest
from knockoff_original.data_gen import SyntheticDataGenerator, GWASDataGenerator
from knockoff_original.decompose import CholeskyDecomposition
from knockoff_original.gram_matrix import compute_gram_matrix, normalize_features
from knockoff_original.knockoff_construct import (
    choose_s_vector,
    generate_knockoff_features,
)


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    generator = SyntheticDataGenerator(n=10000, p=10, noise_variance=5.0)
    X, _ = generator.generate_data()
    return X


@pytest.fixture
def gwas_data():
    np.random.seed(42)
    base_generator = SyntheticDataGenerator(n=10000, p=25, noise_variance=4.0)
    gwas_generator = GWASDataGenerator(base_generator)
    X, _ = gwas_generator.generate_data()
    return X


@pytest.fixture
def sample_matrix(synthetic_data):
    return compute_gram_matrix(normalize_features(synthetic_data))


def test_s_vector_nonnegative(sample_matrix):
    s = choose_s_vector(sample_matrix)
    assert np.all(s >= 0), "s should be non-negative"


def test_s_vector_upper_bound(sample_matrix):
    tau = 1.5
    s = choose_s_vector(sample_matrix, tau=tau)
    assert np.all(s <= tau), "s should not exceed the specified tau"


def test_knockoff_features_dimensions(synthetic_data):
    s = np.full(synthetic_data.shape[1], 0.5)
    X_tilde = generate_knockoff_features(synthetic_data, s)
    assert (
        X_tilde.shape == synthetic_data.shape
    ), "Dimensions of knockoff features must match original"


def test_knockoff_features_statistical_properties(gwas_data):
    X = normalize_features(gwas_data)
    s = choose_s_vector(compute_gram_matrix(X), 0.1)
    X_tilde = generate_knockoff_features(X, s, normalize=False)
    original_mean = np.mean(X, axis=0)
    knockoff_mean = np.mean(X_tilde, axis=0)
    assert np.allclose(
        original_mean, knockoff_mean, atol=1e-1
    ), "Means of original and knockoff should be close (within tolerance)"


def test_stability_of_knockoff_generation(synthetic_data):
    X = normalize_features(synthetic_data)
    s = choose_s_vector(compute_gram_matrix(X), 0.5)

    first_run = generate_knockoff_features(X, s, normalize=False)
    second_run = generate_knockoff_features(X, s, normalize=False)
    third_run = generate_knockoff_features(X, s, normalize=False)

    assert np.allclose(
        np.mean(first_run, axis=0),
        np.mean(second_run, axis=0),
        np.mean(third_run, axis=0),
        atol=1e-1,
    ), "Knockoff generation should be stable (up to a tolerance)"


def test_sensitivity_of_s_in_knockoff_generation(gwas_data):
    X_gwas = gwas_data
    s_low = np.full(X_gwas.shape[1], 0.1)
    s_high = np.full(X_gwas.shape[1], 0.9)
    knockoff_low = generate_knockoff_features(X_gwas, s_low)
    knockoff_high = generate_knockoff_features(X_gwas, s_high)
    correlation_low = np.corrcoef(X_gwas.ravel(), knockoff_low.ravel())[0, 1]
    correlation_high = np.corrcoef(X_gwas.ravel(), knockoff_high.ravel())[0, 1]
    assert (
        correlation_low > correlation_high
    ), "Higher s should reduce correlation between original and knockoff"
