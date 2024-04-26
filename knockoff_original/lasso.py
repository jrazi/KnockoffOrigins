import numpy as np
from sklearn.linear_model import Lasso


def prepare_augmented_design_matrix(X, X_tilde):
    """
    Concatenate original and knockoff features to create the augmented design matrix.

    Args:
        X: Original feature matrix (n x p).
        X_tilde: Knockoff feature matrix (n x p).

    Returns:
        Augmented design matrix (n x 2p).
    """
    return np.concatenate([X, X_tilde], axis=1)


def fit_lasso(X, y, alpha):
    """
    Fit Lasso regression model.

    Args:
        X: Feature matrix (n x p).
        y: Response vector (n x 1).
        alpha: Regularization parameter.

    Returns:
        Fitted Lasso model.
    """
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    return model


def compute_feature_importance(X, X_tilde, y, alpha):
    """
    Compute feature importance statistic W_j.

    Args:
        X: Original feature matrix (n x p).
        X_tilde: Knockoff feature matrix (n x p).
        y: Response vector (n x 1).
        alpha: Regularization parameter.

    Returns:
        Feature importance vector W (p x 1).
    """
    # Create augmented design matrix
    X_augmented = prepare_augmented_design_matrix(X, X_tilde)

    # Fit Lasso model to augmented matrix
    lasso_model = fit_lasso(X_augmented, y, alpha)

    # Extract coefficients and compute feature importance
    coefficients = lasso_model.coef_
    n_features = X.shape[1]
    W = np.abs(coefficients[:n_features]) - np.abs(coefficients[n_features:])

    return W
