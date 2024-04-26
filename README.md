# 🔍 KnockoffOrigins: An Implementation of "CONTROLLING THE FALSE DISCOVERY RATE VIA KNOCKOFFS (2015)"

[![Python package](https://github.com/jrazi/KnockoffOrigins/actions/workflows/python-package.yml/badge.svg?event=registry_package)](https://github.com/jrazi/KnockoffOrigins/actions/workflows/python-package.yml)


This repository hosts the implementation of the knockoff filter method for controlled variable selection, based on the "Controlling the False Discovery Rate via Knockoffs" paper from 2015. The method is designed for high-dimensional data settings to effectively control the false discovery rate while preserving statistical power.

**Note:** Much of this implementation was crafted either from scratch or without relying on high-level libraries. This approach was chosen primarily for educational purposes (mostly _self-educational_ purposes), allowing for a deeper understanding and exploration of the underlying algorithms. While this method involves some "reinventing the wheel," it might have some educational value. Future development may include the integration of more specialized libraries to enhance functionality and performance.

## Table of Content

- [🔍 KnockoffOrigins: An Implementation of "CONTROLLING THE FALSE DISCOVERY RATE VIA KNOCKOFFS (2015)"](#-knockofforigins-an-implementation-of-controlling-the-false-discovery-rate-via-knockoffs-2015)
  - [Table of Content](#table-of-content)
  - [Features](#features)
  - [Installation](#installation)
    - [Using Pip](#using-pip)
    - [Using Poetry](#using-poetry)
    - [From Source](#from-source)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)
  - [TODO](#todo)

## Features

- Implementation of the knockoff filter method for feature selection.
- Synthetic and GWAS-like data generators for evaluation and testing.
- Lasso regression integration for feature importance assessment.

## Installation

`KnockoffOrigins` is available on PyPI and can be installed using either pip or Poetry.

### Using Pip

You can install `KnockoffOrigins` directly using pip:

```bash
pip install KnockoffOrigins
```

This command will download and install the latest version of KnockoffOrigins along with its dependencies.
Using Poetry

### Using Poetry

If you are using Poetry for your project, you can add KnockoffOrigins to your project as follows:

```bash
poetry add KnockoffOrigins
```

This will handle the installation and also update your pyproject.toml and poetry.lock files to reflect the change.

### From Source

If you prefer to install from source or want to contribute to the package, first ensure Poetry dependency management is installed:

```bash
pip install poetry
```

Then clone the repository and install the dependencies:

```bash
git clone https://github.com/jrazi/KnockoffOrigins.git
cd KnockoffOrigins
poetry install
```

## Usage

### Generating Knockoff Features

To generate knockoff features based on your original data, you can use the `generate_knockoff_features` function from the `knockofforigins.knockoff_construct` module.

```python
import numpy as np
from knockofforigins.knockoff_construct import generate_knockoff_features, choose_s_vector

# Load your original feature matrix X (n x p)
# ...

# Choose the s vector for knockoff construction
s = choose_s_vector(np.cov(X.T))

# Generate knockoff features
X_knockoff = generate_knockoff_features(X, s)
```

### Feature Selection with Lasso

After generating knockoff features, you can perform feature selection using the Lasso regression model with the augmented design matrix (original and knockoff features concatenated).

```python
from knockofforigins.lasso import compute_feature_importance

# Load your response vector y (n x 1)
# ...

# Compute feature importance statistic W
alpha = 0.1  # Regularization parameter for Lasso
W = compute_feature_importance(X, X_knockoff, y, alpha)

# Select features based on W
selected_features = np.argsort(-W)[:num_features_to_select]
```

### Generating Synthetic GWAS Data

If you need to generate synthetic GWAS data for testing purposes, you can use the InfluentialFeatureGWASDataGenerator class.

```python
from KnockoffOrigins.knockoff_construct import generate_knockoff_features, choose_s_vector
from KnockoffOrigins.gram_matrix import generate_gram_matrix

# Calculate the covariance matrix Sigma of the original features X
Sigma = generate_gram_matrix(X)

# Choose the vector 's' for knockoff feature generation
s = choose_s_vector(Sigma)

# Generate knockoff features using the original features X
X_knockoff = generate_knockoff_features(X, s)
```

## Contributing

Contributions are welcome, and appreciated!

## License

This project is licensed under the MIT License.

## TODO

- [ ] Implement test statistics for feature evaluation.
- [ ] Develop FDR control mechanisms as outlined in the original study.
- [ ] Implement Lasso feature selection using lower-level libraries.
- [ ] Address some of the bugs and implementation issues.
- [ ] Create example notebooks demonstrating package usage.
- [ ] Replicate experiments from the original 2015 knockoff paper.
- [ ] Develop visualization methods for feature selection analysis.
- [ ] Implement the KnockOff+ method for enhanced feature selection.
