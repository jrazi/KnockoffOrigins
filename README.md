# 🔍 KnockoffOrigins: An Implementation of "CONTROLLING THE FALSE DISCOVERY RATE VIA KNOCKOFFS (2015)"

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

Here is a quick example of how to generate data and apply the knockoff filter:

```python
from KnockOffOrigins.data_gen import SyntheticDataGenerator, GWASDataGenerator

# Initialize the data generator
base_generator = SyntheticDataGenerator(n=10000, p=100, noise_variance=1.0)

# Generate data and apply the knockoff filter
X, y = base_generator.generate_data()
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
