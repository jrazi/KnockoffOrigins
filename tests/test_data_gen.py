import numpy as np
import pytest
from knockofforigins.data_gen import SyntheticDataGenerator, GWASDataGenerator


class TestSyntheticDataGenerator:
    @pytest.fixture
    def generator(self):
        """Fixture to provide a SyntheticDataGenerator instance."""
        return SyntheticDataGenerator(n=100, p=10, noise_variance=1.0)

    def test_initialization(self, generator):
        """Tests the initialization of SyntheticDataGenerator."""
        assert generator.n == 100, "Number of samples (n) not set correctly."
        assert generator.p == 10, "Number of features (p) not set correctly."
        assert generator.noise_variance == 1.0, "Noise variance not set correctly."

    def test_generate_data_shapes(self, generator):
        """Tests the shapes of the generated data by SyntheticDataGenerator."""
        X, y = generator.generate_data()
        assert X.shape == (100, 10), "Incorrect shape for feature matrix (X)."
        assert y.shape == (100,), "Incorrect shape for response vector (y)."

    def test_generate_data_types(self, generator):
        """Tests the data types of the generated data by SyntheticDataGenerator."""
        X, y = generator.generate_data()
        assert isinstance(X, np.ndarray), "X is not a numpy array."
        assert isinstance(y, np.ndarray), "y is not a numpy array."
        assert X.dtype == float, "X should contain floating point values."
        assert y.dtype == float, "y should contain floating point values."


class TestGWASDataGenerator:
    @pytest.fixture
    def base_generator(self):
        """Fixture to provide a SyntheticDataGenerator instance for GWAS data generation."""
        return SyntheticDataGenerator(n=100, p=10, noise_variance=0.5)

    @pytest.fixture
    def gwas_generator(self, base_generator):
        """Fixture to provide a GWASDataGenerator instance."""
        return GWASDataGenerator(base_generator)

    def test_gwas_generator_init(self, gwas_generator, base_generator):
        """Tests the initialization of GWASDataGenerator."""
        assert (
            gwas_generator._base_generator == base_generator
        ), "GWASDataGenerator not initialized correctly."

    def test_gwas_generate_data_shapes(self, gwas_generator):
        """Tests the shapes of the generated GWAS data."""
        X, y = gwas_generator.generate_data()
        assert X.shape == (100, 10), "Incorrect shape for GWAS feature matrix (X)."
        assert y.shape == (100,), "Incorrect shape for GWAS response vector (y)."

    def test_gwas_generate_data_types(self, gwas_generator):
        """Tests the data types of the generated GWAS data."""
        X, y = gwas_generator.generate_data()
        assert isinstance(X, np.ndarray), "X is not a numpy array."
        assert isinstance(y, np.ndarray), "y is not a numpy array."
        assert X.dtype == int, "GWAS features (X) should be integers (0, 1, 2)."
        assert y.dtype == int, "GWAS phenotype (y) should be binary (0, 1)."
