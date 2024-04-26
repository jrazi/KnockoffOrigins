# test_feature_selection.py

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from KnockOffOrigins.lasso import (
    prepare_augmented_design_matrix,
    fit_lasso,
    compute_feature_importance,
)
from KnockOffOrigins.data_gen import (
    InfluentialFeatureGWASDataGenerator,
    SyntheticDataGenerator,
)


@pytest.fixture
def gwas_data():
    """
    Fixture to generate high-dimensional GWAS data with a limited number of influential features.
    """
    base_generator = SyntheticDataGenerator(
        n=200, p=20, noise_variance=1.0
    )  # n is 10 times p
    data_generator = InfluentialFeatureGWASDataGenerator(
        base_generator, num_influential=10
    )
    X, y = data_generator.generate_data()
    return X, y


@pytest.fixture
def gwas_knockoff_data(gwas_data):
    """
    Fixture to generate knockoff data corresponding to the GWAS data.
    """
    X, _ = gwas_data
    X_tilde = np.random.normal(
        loc=0, scale=1, size=X.shape
    )  # Simple knockoff generation
    X_tilde = np.clip(np.round(X_tilde * 2), 0, 2).astype(int)  # Convert to GWAS format
    return X, X_tilde


def test_lasso_with_high_dimensional_data(gwas_data):
    """
    Test Lasso with high-dimensional GWAS data where only a few features are actually influential.
    """
    X, y = gwas_data
    model = fit_lasso(X, y, alpha=0.1)
    assert len(model.coef_) == X.shape[1], "Incorrect number of coefficients"
    assert np.any(model.coef_ != 0), "Expected some non-zero coefficients"


def test_feature_importance_with_knockoff(gwas_knockoff_data):
    """
    Test feature selection using Lasso with original and knockoff data to distinguish important features.
    """
    X, X_tilde = gwas_knockoff_data
    y = (
        X[:, :10] @ np.ones(10)
    ) > 5  # Generate a binary response based on the first 10 features
    y = y.astype(int)
    W = compute_feature_importance(X, X_tilde, y, alpha=0.05)
    assert (
        len(W) == X.shape[1]
    ), "Feature importance vector W should match the number of original features"
    assert np.any(W != 0), "Expected some non-zero importance scores"


@pytest.mark.parametrize("alpha", [0.01, 0.1, 1, 10])
def test_lasso_alpha_sensitivity(gwas_data, alpha):
    """
    Test the sensitivity of the Lasso model to changes in the regularization parameter alpha.
    """
    X, y = gwas_data
    model = fit_lasso(X, y, alpha)
    non_zero_coefs = np.count_nonzero(model.coef_)
    assert non_zero_coefs <= X.shape[1], "Too many non-zero coefficients"
    # Expect fewer non-zero coefficients with increasing alpha
    if alpha >= 1:
        assert (
            non_zero_coefs < 10
        ), "Expecting fewer non-zero coefficients for higher alpha values"


def test_lasso_convergence_with_high_dimensions(gwas_data):
    """
    Test if Lasso converges properly with high-dimensional data.
    """
    X, y = gwas_data
    model = fit_lasso(X, y, alpha=1e-10)
    assert (
        model.n_iter_ < model.max_iter
    ), "Model should converge before reaching the maximum number of iterations"
