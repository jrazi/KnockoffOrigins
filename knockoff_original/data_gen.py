import numpy as np


class SyntheticDataGenerator:
    def __init__(self, n: int, p: int, noise_variance: float):
        """
        Initialize the data generator.

        Parameters:
        - n (int): Number of samples (rows).
        - p (int): Number of features (columns).
        - noise_variance (float): Variance of the Gaussian noise in the model.
        """
        self.n = n
        self.p = p
        self.noise_variance = noise_variance

    def generate_data(self):
        """
        Generate synthetic data based on a Gaussian linear model.

        Returns:
        - X (ndarray): Feature matrix.
        - y (ndarray): Response vector.
        """
        X = np.random.normal(0, 1, (self.n, self.p))
        beta = np.random.normal(0, 1, self.p)
        noise = np.random.normal(0, np.sqrt(self.noise_variance), self.n)
        y = X @ beta + noise
        return X, y


class GWASDataGenerator(SyntheticDataGenerator):
    def __init__(self, base_generator: SyntheticDataGenerator):
        """
        Initialize the GWAS data generator.

        Parameters:
        - base_generator (SyntheticDataGenerator): The base data generator used for data generation.
        """
        self.base_generator = base_generator

    def generate_data(self):
        """
        Generate synthetic GWAS data.

        Returns:
        - X (ndarray): GWAS feature matrix (integers 0, 1, 2).
        - y (ndarray): Binary response vector.
        """
        X, y_continuous = self.base_generator.generate_data()
        X_gwas = np.clip(np.round(X * 2), 0, 2).astype(int)
        y_binary = (y_continuous > np.median(y_continuous)).astype(int)
        return X_gwas, y_binary


import numpy as np


class InfluentialFeatureGWASDataGenerator(GWASDataGenerator):
    def __init__(
        self, base_generator: SyntheticDataGenerator, num_influential: int = 10
    ):
        """
        Extend GWASDataGenerator to generate data with a limited number of influential features.

        Parameters:
        - base_generator (SyntheticDataGenerator): The base data generator.
        - num_influential (int): Number of features that are actually influential.
        """
        self.base_generator = base_generator
        self.num_influential = min(
            num_influential, base_generator.p
        )  # Cap at total number of features

    def generate_data(self):
        """
        Generate GWAS data with a specified number of features that truly influence the response.

        Returns:
        - X (ndarray): GWAS feature matrix (integers 0, 1, 2).
        - y (ndarray): Binary response vector, semi-balanced.
        """
        X, _ = self.base_generator.generate_data()
        beta = np.zeros(self.base_generator.p)

        # Set coefficients for a subset of features to be non-zero
        influential_indices = np.random.choice(
            self.base_generator.p, self.num_influential, replace=False
        )
        beta[influential_indices] = np.random.normal(0, 1, self.num_influential)

        # Generate the response using a logistic model
        linear_combination = X @ beta
        probability = 1 / (1 + np.exp(-linear_combination))
        y_binary = np.random.binomial(1, probability, self.base_generator.n)

        if np.mean(y_binary) < 0.45 or np.mean(y_binary) > 0.55:
            threshold = np.percentile(probability, 50)
            y_binary = (probability > threshold).astype(int)

        X_gwas = np.clip(np.round(X * 2), 0, 2).astype(int)  # Convert to GWAS format
        return X_gwas, y_binary
