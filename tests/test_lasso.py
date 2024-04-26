import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from KnockOffOrigins.lasso import (
    prepare_augmented_design_matrix,
    fit_lasso,
    compute_feature_importance,
)


@pytest.fixture
def example_matrices():
    X = np.array([[1, 2], [3, 4]])
    X_tilde = np.array([[2, 1], [4, 3]])
    return X, X_tilde


@pytest.fixture
def regression_setup():
    X = np.random.randn(100, 10)
    X_tilde = np.random.randn(100, 10)
    y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100)  # response variable with noise
    alpha = 0.1
    return X, X_tilde, y, alpha


def test_augmented_design_matrix(example_matrices):
    X, X_tilde = example_matrices
    X_augmented = prepare_augmented_design_matrix(X, X_tilde)
    assert X_augmented.shape == (2, 4)  # Check the shape
    assert np.array_equal(X_augmented[:, :2], X)  # Check the first part of the matrix
    assert np.array_equal(
        X_augmented[:, 2:], X_tilde
    )  # Check the second part of the matrix


def test_empty_matrices():
    X_empty = np.array([[]])
    X_tilde_empty = np.array([[]])
    X_augmented = prepare_augmented_design_matrix(X_empty, X_tilde_empty)
    assert X_augmented.size == 0  # Expect empty output


def test_fit_lasso(regression_setup):
    X, X_tilde, y, alpha = regression_setup
    model = fit_lasso(X, y, alpha)
    assert model.intercept_ is not None  # Model has an intercept
    assert len(model.coef_) == X.shape[1]  # Coefficients array is correct length


def test_high_alpha(regression_setup):
    X, X_tilde, y, _ = regression_setup
    high_alpha = 100  # Large enough to shrink all coefficients to zero
    model = fit_lasso(X, y, high_alpha)
    assert np.allclose(
        model.coef_, np.zeros(X.shape[1]), atol=1e-2
    )  # Check coefficients are all zero


def test_compute_feature_importance(regression_setup):
    X, X_tilde, y, alpha = regression_setup
    W = compute_feature_importance(X, X_tilde, y, alpha)
    assert len(W) == X.shape[1]  # Correct length of the importance vector
    assert np.any(W != 0)  # Check that some importance values are non-zero


def test_zero_importance_with_high_alpha(regression_setup):
    X, X_tilde, y, _ = regression_setup
    high_alpha = 100
    W = compute_feature_importance(X, X_tilde, y, high_alpha)
    assert np.allclose(
        W, np.zeros(X.shape[1]), atol=1e-2
    )  # Expect zero importance due to high alpha
