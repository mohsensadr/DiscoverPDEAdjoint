[![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.2401.17177.svg)](https://doi.org/10.48550/arXiv.2401.17177)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# Discover PDEs using the Adjoint Method

This git repository contains a Python implementation of the adjoint method for discovering PDEs given data. The arxiv version of the corresponding manuscript can be found here:
https://doi.org/10.48550/arXiv.2401.17177


# Usage

To use this library, import the content of ```src/``` directory via

```
import sys
import os
src_path = os.path.abspath(os.path.join(os.getcwd(), '../src'))
sys.path.append(src_path)
from adjoint import *
```

Given that the solution of PDE ```f``` discretized on a temporal ```t``` and spatial grid ```x```, is stored in a NumPy array with dimension

```(num of PDEs, Number of time steps, number of grid points in x1, number of grid points in x2, ...)```

the adjoint solver can be called simply by

```
estimated_params, eps, losses = AdjointFindPDE(f, x, dx, data_dt=dt)
```

For more details, see notebooks in ```examples/```.
